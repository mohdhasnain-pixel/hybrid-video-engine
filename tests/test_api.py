import time

from fastapi.testclient import TestClient


def wait(client, job_id, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["state"] in ("COMPLETED", "FAILED", "CANCELLED"):
            return j
        time.sleep(0.5)
    raise TimeoutError(j["state"])


def test_upload_process_compare_delete(clips):
    from video_engine.server import app

    with TestClient(app) as client:
        cfg = client.get("/api/config").json()
        assert cfg["storage"]["backend"] == "local"
        assert [p["name"] for p in cfg["presets"]][:2] == ["fast", "balanced"]
        assert "<title>Video Enhancer</title>" in client.get("/").text

        with open(clips["small30"], "rb") as f:
            r = client.post("/api/jobs", files={"file": ("my clip.mp4", f, "video/mp4")},
                            data={"preset": "fast", "max_seconds": "1"})
        assert r.status_code == 201, r.text
        job = wait(client, r.json()["id"])
        assert job["state"] == "COMPLETED", job.get("error")
        assert job["validation"]["passed"]
        sc = job["size_comparison"]
        assert sc["output_bytes"] > 0 and sc["difference_bytes"] == sc["output_bytes"] - sc["input_bytes"]
        assert {"source_preview.mp4", "output_preview.mp4", "poster_source.jpg", "poster_output.jpg"} <= set(job["comparison"]["urls"])
        assert len(job["comparison"]["crops"]) == 2

        video = client.get(job["output"]["url"], headers={"Range": "bytes=0-99"})
        assert video.status_code == 206 and len(video.content) == 100
        dl = client.get(job["output"]["download_url"])
        assert "attachment" in dl.headers["content-disposition"]

        assert client.delete(f"/api/jobs/{job['id']}").json()["deleted"]
        assert client.get(f"/api/jobs/{job['id']}").status_code == 404
        assert client.get(job["output"]["url"]).status_code == 404


def test_rejects_bad_preset_and_empty_upload(clips):
    from video_engine.server import app

    with TestClient(app) as client:
        assert client.post("/api/jobs", files={"file": ("a.mp4", b"x")}, data={"preset": "nope"}).status_code == 422
        assert client.post("/api/jobs", files={"file": ("a.mp4", b"")}).status_code == 422


def test_invalid_video_fails_cleanly(clips):
    from video_engine.server import app

    with TestClient(app) as client:
        r = client.post("/api/jobs", files={"file": ("bad.mp4", b"not a video at all" * 100)}, data={"preset": "fast"})
        job = wait(client, r.json()["id"])
        assert job["state"] == "FAILED"
        assert job["error"]["stage"] == "ANALYZING"


def test_full_job_in_s3_mode(clips, monkeypatch, tmp_path):
    import boto3
    from moto import mock_aws

    from video_engine import config, storage

    with mock_aws():
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
        monkeypatch.setenv("VE_STORAGE_BACKEND", "s3")
        monkeypatch.setenv("VE_S3_BUCKET", "videos")
        monkeypatch.setenv("VE_S3_REGION", "us-east-1")
        config.get_settings.cache_clear()
        storage.get_storage.cache_clear()
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="videos")

        from video_engine.server import app

        with TestClient(app) as client:
            assert client.get("/api/config").json()["storage"]["backend"] == "s3"
            with open(clips["small30"], "rb") as f:
                r = client.post("/api/jobs", files={"file": ("clip.mp4", f)}, data={"preset": "fast"})
            job = wait(client, r.json()["id"])
            assert job["state"] == "COMPLETED", job.get("error")
            assert job["storage_backend"] == "s3"
            assert job["output"]["url"].startswith("https://videos.s3")
            keys = {o["Key"] for o in s3.list_objects_v2(Bucket="videos")["Contents"]}
            jid = job["id"]
            assert f"video-engine/jobs/{jid}/input/clip.mp4" in keys
            assert f"video-engine/jobs/{jid}/output/clip_enhanced.mp4" in keys
            assert f"video-engine/jobs/{jid}/compare/source_preview.mp4" in keys
            # nothing left on local disk except job metadata
            assert not any((tmp_path / "work").glob(f"{jid}/*"))
            assert not (tmp_path / "storage" / "jobs").exists()
            client.delete(f"/api/jobs/{jid}")
            assert "Contents" not in s3.list_objects_v2(Bucket="videos")


def test_estimate_endpoint():
    from video_engine.server import app

    with TestClient(app) as client:
        e = client.get("/api/estimate", params={"width": 1080, "height": 1920, "duration": 60}).json()
        assert e["seconds"] > 0 and e["target"] == "2988x5312" and e["frames_out"] == 3600
        short = client.get("/api/estimate", params={"width": 1080, "height": 1920, "duration": 60,
                                                    "max_seconds": 5}).json()
        assert short["seconds"] < e["seconds"]
        assert client.get("/api/estimate", params={"width": 0, "height": 1, "duration": 1}).status_code == 422
