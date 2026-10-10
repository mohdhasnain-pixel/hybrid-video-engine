"""Model weights: where they come from, their licences, pinned checksums.

Weights are kept out of the source tree (``models/``) and downloaded with
``video-engine download-models``. Every job records the model name, version
and checksum it used.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .config import get_settings


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str  # super_resolution / interpolation
    scale: int
    files: dict[str, tuple[str, str]]  # local filename -> (url, sha256)
    license: str
    source: str
    description: str
    zip_member: str | None = None  # for weights shipped inside a zip


REGISTRY: dict[str, ModelSpec] = {
    "realesr-general-x4v3": ModelSpec(
        name="realesr-general-x4v3",
        kind="super_resolution",
        scale=4,
        files={
            "realesr-general-x4v3.pth": (
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
            ),
            "realesr-general-wdn-x4v3.pth": (
                "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
                "1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca",
            ),
        },
        license="BSD-3-Clause (Real-ESRGAN, Xintao Wang et al.)",
        source="https://github.com/xinntao/Real-ESRGAN",
        description="Compact SRVGG 4× real-world SR (1.2M params) with DNI denoise-strength control.",
    ),
    "rife-v4.26": ModelSpec(
        name="rife-v4.26",
        kind="interpolation",
        scale=1,
        files={
            "rife-v4.26-flownet.pkl": (
                "https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip",
                "45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f",
            ),
        },
        zip_member="RIFEv4.26_0921/flownet.pkl",
        license="MIT (Practical-RIFE, hzwer)",
        source="https://github.com/hzwer/Practical-RIFE",
        description="Real-time intermediate flow estimation, arbitrary timestep.",
    ),
}


def models_dir() -> Path:
    return get_settings().models_dir


def get(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model {name!r}")
    return REGISTRY[name]


def path_for(name: str, filename: str | None = None) -> Path:
    spec = get(name)
    return models_dir() / (filename or next(iter(spec.files)))


def is_installed(name: str) -> bool:
    if name not in REGISTRY:
        return False
    return all((models_dir() / f).is_file() for f in REGISTRY[name].files)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest() -> dict:
    p = models_dir() / "manifest.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def model_info(name: str) -> dict:
    spec = get(name)
    m = manifest()
    return {
        "name": name,
        "license": spec.license,
        "source": spec.source,
        "files": {f: m.get(f, {}).get("sha256") for f in spec.files},
    }


def download(name: str, log=print) -> None:
    spec = get(name)
    d = models_dir()
    d.mkdir(parents=True, exist_ok=True)
    m = manifest()
    for filename, (url, expected) in spec.files.items():
        dest = d / filename
        if dest.is_file():
            log(f"  {filename}: present")
        else:
            log(f"  {filename}: downloading {url}")
            with urllib.request.urlopen(url, timeout=120) as r:
                payload = r.read()
            if spec.zip_member:
                payload = zipfile.ZipFile(io.BytesIO(payload)).read(spec.zip_member)
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(payload)
            tmp.rename(dest)
        digest = sha256(dest)
        if expected and digest != expected:
            log(f"  WARNING {filename}: sha256 {digest} does not match pinned {expected}")
        m[filename] = {"model": name, "sha256": digest, "url": url, "license": spec.license}
    (d / "manifest.json").write_text(json.dumps(m, indent=2))


def status() -> dict:
    return {name: {"installed": is_installed(name), "kind": s.kind, "license": s.license,
                   "description": s.description} for name, s in REGISTRY.items()}
