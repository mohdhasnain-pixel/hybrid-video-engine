"""Phase 11 — jobs: state machine, persistence, background worker, cleanup.

    QUEUED → ANALYZING → PLANNING → PROCESSING → VALIDATING → STORING → COMPLETED
                                       ↘ FAILED / CANCELLED (stage + error recorded)

Jobs run one at a time (the GPU is the bottleneck; concurrent AI stages would
only compete for unified memory). Job metadata is JSON on local disk; the video
files themselves go to the storage backend selected by the feature flag.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from . import compare
from .config import get_settings
from .engine import EngineError, enhance_video
from .media import MediaError
from .pipeline import Cancelled
from .planner import JobOptions
from .storage import Storage, get_storage

log = logging.getLogger(__name__)

ACTIVE_STATES = {"QUEUED", "ANALYZING", "PLANNING", "PROCESSING", "VALIDATING", "STORING"}


class Job(BaseModel):
    id: str
    created_at: float
    updated_at: float
    state: str = "QUEUED"
    storage_backend: str
    options: JobOptions
    input: dict  # filename, key, size_bytes
    output: dict | None = None  # key, size_bytes, filename
    progress: dict = Field(default_factory=dict)
    profile: dict | None = None
    degradation: dict | None = None
    plan: dict | None = None
    metrics: dict | None = None
    validation: dict | None = None
    timings: dict = Field(default_factory=dict)
    comparison: dict | None = None
    error: dict | None = None
    history: list[dict] = Field(default_factory=list)


class JobStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}
        for p in root.glob("*.json"):
            try:
                job = Job.model_validate_json(p.read_text())
            except Exception:  # noqa: BLE001 - skip unreadable records
                log.warning("Skipping unreadable job file %s", p)
                continue
            self._jobs[job.id] = job
            if job.state in ACTIVE_STATES:  # the server stopped mid-job
                job.error = {"stage": job.state, "message": "Interrupted: the server restarted while this job ran."}
                job.state = "FAILED"
                self.save(job)

    def save(self, job: Job) -> None:
        with self._lock:
            job.updated_at = time.time()
            self._jobs[job.id] = job
            tmp = self.root / f"{job.id}.json.tmp"
            tmp.write_text(job.model_dump_json(indent=1))
            tmp.replace(self.root / f"{job.id}.json")

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
            (self.root / f"{job_id}.json").unlink(missing_ok=True)


class JobManager:
    def __init__(self, store: JobStore | None = None):
        s = get_settings()
        self.store = store or JobStore(s.jobs_dir)
        self.queue: queue.Queue[str] = queue.Queue()
        self.cancel_events: dict[str, threading.Event] = {}
        self._thread = threading.Thread(target=self._loop, name="ve-job-worker", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------- lifecycle
    def submit(self, upload_path: Path, filename: str, options: JobOptions) -> Job:
        s = get_settings()
        storage = get_storage()
        job_id = uuid.uuid4().hex[:12]
        safe = Path(filename).name or "input.mp4"
        work = s.work_dir / job_id
        work.mkdir(parents=True, exist_ok=True)
        key = f"jobs/{job_id}/input/{safe}"
        if storage.name == "local":
            size = storage.put_file(upload_path, key, move=True)  # processed in place, no extra copy
        else:
            # keep a local copy for processing; the worker uploads it to S3 before analysis
            local_input = work / f"input{Path(safe).suffix.lower() or '.mp4'}"
            shutil.move(str(upload_path), local_input)
            size = local_input.stat().st_size
        now = time.time()
        job = Job(id=job_id, created_at=now, updated_at=now, storage_backend=storage.name,
                  options=options, input={"filename": safe, "key": key, "size_bytes": size})
        self._transition(job, "QUEUED")
        self.cancel_events[job_id] = threading.Event()
        self.queue.put(job_id)
        return job

    def cancel(self, job_id: str) -> bool:
        ev = self.cancel_events.get(job_id)
        job = self.store.get(job_id)
        if not job or job.state not in ACTIVE_STATES:
            return False
        if ev:
            ev.set()
        if job.state == "QUEUED":
            self._transition(job, "CANCELLED")
        return True

    def delete(self, job_id: str) -> bool:
        job = self.store.get(job_id)
        if not job:
            return False
        if job.state in ACTIVE_STATES:
            self.cancel(job_id)
        try:
            get_storage(job.storage_backend).delete_prefix(f"jobs/{job_id}")
        except Exception as exc:  # noqa: BLE001 - metadata removal must still succeed
            log.warning("Could not delete stored files for %s: %s", job_id, exc)
        shutil.rmtree(get_settings().work_dir / job_id, ignore_errors=True)
        self.store.delete(job_id)
        return True

    def _transition(self, job: Job, state: str, **fields) -> None:
        job.state = state
        for k, v in fields.items():
            setattr(job, k, v)
        job.history.append({"state": state, "at": time.time()})
        self.store.save(job)

    # ---------------------------------------------------------------- worker
    def _loop(self) -> None:
        while True:
            job_id = self.queue.get()
            job = self.store.get(job_id)
            if job is None or job.state != "QUEUED":
                continue
            try:
                self._run(job)
            except Exception:  # noqa: BLE001 - never let the worker die
                log.exception("Unhandled error in job %s", job_id)

    def _run(self, job: Job) -> None:
        s = get_settings()
        storage = get_storage(job.storage_backend)
        cancel = self.cancel_events.setdefault(job.id, threading.Event())
        work = s.work_dir / job.id
        work.mkdir(parents=True, exist_ok=True)
        out_name = f"{Path(job.input['filename']).stem}_enhanced.mp4"
        local_output = work / out_name
        last_save = [0.0]

        def on_state(state: str, data: dict | None) -> None:
            if data:
                for k, v in data.items():
                    setattr(job, k, v)
            if state != job.state:
                self._transition(job, state)
            else:
                self.store.save(job)

        def on_progress(p: dict) -> None:
            job.progress = p
            if time.time() - last_save[0] > 1.0:
                last_save[0] = time.time()
                self.store.save(job)

        stage = "ANALYZING"
        try:
            local_input = work / f"input{Path(job.input['filename']).suffix.lower() or '.mp4'}"
            if local_input.is_file():
                if not storage.exists(job.input["key"]):
                    storage.put_file(local_input, job.input["key"])
            else:
                local_input = storage.get_file(job.input["key"], local_input)  # local: the stored file itself
            result = enhance_video(local_input, local_output, job.options, work, on_state, on_progress, cancel)
            job.metrics, job.validation, job.timings = result["metrics"], result["validation"], result["timings"]
            stage = "STORING"
            self._transition(job, "STORING")
            job.comparison = self._comparison(job, local_input, local_output, work, storage)
            out_key = f"jobs/{job.id}/output/{out_name}"
            size = storage.put_file(local_output, out_key, move=True)
            job.output = {"key": out_key, "size_bytes": size, "filename": out_name}
            job.progress = {**job.progress, "fraction": 1.0, "eta_seconds": 0}
            self._transition(job, "COMPLETED")
        except Cancelled:
            self._transition(job, "CANCELLED")
        except EngineError as exc:
            self._fail(job, exc.stage, str(exc), exc.detail)
        except MediaError as exc:
            self._fail(job, job.state, str(exc), exc.stderr[-4000:])
        except Exception as exc:  # noqa: BLE001
            self._fail(job, job.state if job.state != "QUEUED" else stage, f"{type(exc).__name__}: {exc}",
                       traceback.format_exc()[-4000:])
        finally:
            if not s.keep_work_files:
                shutil.rmtree(work, ignore_errors=True)
            self.cancel_events.pop(job.id, None)

    def _fail(self, job: Job, stage: str, message: str, detail: str = "") -> None:
        log.error("Job %s failed at %s: %s", job.id, stage, message)
        self._transition(job, "FAILED", error={"stage": stage, "message": message, "detail": detail})

    def _comparison(self, job: Job, src: Path, out: Path, work: Path, storage: Storage) -> dict:
        from .planner import JobPlan

        plan = JobPlan.from_dict(job.plan)
        cdir = work / "compare"
        cdir.mkdir(exist_ok=True)
        result: dict = {"crops": [], "files": {}}
        try:
            compare.source_preview(src, plan, cdir / "source_preview.mp4")
            compare.output_preview(out, plan, cdir / "output_preview.mp4")
            compare.poster(cdir / "source_preview.mp4", cdir / "poster_source.jpg")
            compare.poster(cdir / "output_preview.mp4", cdir / "poster_output.jpg")
            result["crops"] = compare.still_crops(src, out, plan, cdir)
        except Exception as exc:  # noqa: BLE001 - comparison is best-effort, output is still valid
            log.warning("Comparison assets failed for %s: %s", job.id, exc)
            result["error"] = str(exc)
        for f in sorted(cdir.iterdir()):
            key = f"jobs/{job.id}/compare/{f.name}"
            storage.put_file(f, key, move=True)
            result["files"][f.name] = key
        return result


# --------------------------------------------------------------------------- view


def job_view(job: Job) -> dict:
    """Job as JSON for the UI, with fresh URLs from the job's storage backend."""
    d = job.model_dump()
    try:
        storage = get_storage(job.storage_backend)
    except Exception as exc:  # noqa: BLE001 - e.g. S3 job viewed after switching flag back without creds
        d["storage_error"] = str(exc)
        return d
    d["input"]["url"] = storage.url(job.input["key"])
    if job.output:
        d["output"]["url"] = storage.url(job.output["key"])
        d["output"]["download_url"] = storage.url(job.output["key"], job.output["filename"])
        insize, outsize = job.input["size_bytes"], job.output["size_bytes"]
        d["size_comparison"] = {
            "input_bytes": insize, "output_bytes": outsize, "difference_bytes": outsize - insize,
            "ratio": round(outsize / insize, 3) if insize else None,
        }
    if job.comparison:
        d["comparison"]["urls"] = {name: storage.url(key) for name, key in job.comparison.get("files", {}).items()}
    return d


def dump(obj) -> str:
    return json.dumps(obj, indent=1, default=str)
