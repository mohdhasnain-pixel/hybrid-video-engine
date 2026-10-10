"""Video storage behind a feature flag.

``VE_STORAGE_BACKEND=local`` (default) keeps uploads, outputs and comparison
assets on this computer under ``VE_LOCAL_STORAGE_DIR``.
``VE_STORAGE_BACKEND=s3`` keeps them in ``s3://VE_S3_BUCKET/VE_S3_PREFIX``
and serves them to the browser through presigned URLs.

Each job records which backend stored its files, so jobs created before the
flag was switched remain viewable as long as their storage still exists.
"""

from __future__ import annotations

import mimetypes
import shutil
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from .config import Settings, get_settings


class StorageError(RuntimeError):
    pass


class Storage(ABC):
    name: str

    @abstractmethod
    def put_file(self, local_path: Path, key: str, move: bool = False) -> int:
        """Store a file under ``key``; returns its size in bytes."""

    @abstractmethod
    def get_file(self, key: str, local_path: Path) -> Path:
        """Materialise ``key`` as a local file (download for remote backends)."""

    @abstractmethod
    def url(self, key: str, download_name: str | None = None) -> str:
        """URL the browser can use to stream or download ``key``."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def size(self, key: str) -> int: ...

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None: ...

    @abstractmethod
    def describe(self) -> dict: ...


def _content_type(key: str) -> str:
    return mimetypes.guess_type(key)[0] or "application/octet-stream"


class LocalStorage(Storage):
    name = "local"

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if not p.is_relative_to(self.root):
            raise StorageError(f"Invalid key {key!r}")
        return p

    def put_file(self, local_path: Path, key: str, move: bool = False) -> int:
        dest = self.path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(local_path).resolve() != dest:
            (shutil.move if move else shutil.copyfile)(str(local_path), str(dest))
        return dest.stat().st_size

    def get_file(self, key: str, local_path: Path) -> Path:
        src = self.path(key)
        if not src.is_file():
            raise StorageError(f"Missing {key}")
        return src  # already local: no copy needed

    def url(self, key: str, download_name: str | None = None) -> str:
        u = f"/files/{quote(key)}"
        return u + (f"?download={quote(download_name)}" if download_name else "")

    def exists(self, key: str) -> bool:
        return self.path(key).is_file()

    def size(self, key: str) -> int:
        return self.path(key).stat().st_size

    def delete_prefix(self, prefix: str) -> None:
        p = self.path(prefix)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.is_file():
            p.unlink()

    def describe(self) -> dict:
        usage = shutil.disk_usage(self.root)
        used = sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
        return {"backend": "local", "location": str(self.root), "used_bytes": used,
                "free_bytes": usage.free}


class S3Storage(Storage):
    name = "s3"

    def __init__(self, settings: Settings, client=None):
        if not settings.s3_bucket:
            raise StorageError("VE_STORAGE_BACKEND=s3 requires VE_S3_BUCKET")
        import boto3

        self.bucket = settings.s3_bucket
        self.prefix = settings.s3_prefix.strip("/") + "/" if settings.s3_prefix.strip("/") else ""
        self.expiry = settings.s3_presign_expiry_seconds
        self.region = settings.s3_region
        from botocore.config import Config

        # SigV4 presigned URLs: required by all newer AWS regions
        self.client = client or boto3.client("s3", region_name=settings.s3_region,
                                             endpoint_url=settings.s3_endpoint_url,
                                             config=Config(signature_version="s3v4"))

    def _k(self, key: str) -> str:
        return self.prefix + key

    def put_file(self, local_path: Path, key: str, move: bool = False) -> int:
        size = Path(local_path).stat().st_size
        self.client.upload_file(str(local_path), self.bucket, self._k(key),
                                ExtraArgs={"ContentType": _content_type(key)})
        if move:
            Path(local_path).unlink(missing_ok=True)
        return size

    def get_file(self, key: str, local_path: Path) -> Path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, self._k(key), str(local_path))
        return local_path

    def url(self, key: str, download_name: str | None = None) -> str:
        params = {"Bucket": self.bucket, "Key": self._k(key)}
        if download_name:
            params["ResponseContentDisposition"] = f'attachment; filename="{download_name}"'
        return self.client.generate_presigned_url("get_object", Params=params, ExpiresIn=self.expiry)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._k(key))
            return True
        except ClientError:
            return False

    def size(self, key: str) -> int:
        return int(self.client.head_object(Bucket=self.bucket, Key=self._k(key))["ContentLength"])

    def delete_prefix(self, prefix: str) -> None:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._k(prefix)):
            objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if objs:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": objs})

    def describe(self) -> dict:
        return {"backend": "s3", "location": f"s3://{self.bucket}/{self.prefix}", "region": self.region}


@lru_cache
def get_storage(name: str | None = None) -> Storage:
    s = get_settings()
    name = name or s.storage_backend
    if name == "local":
        return LocalStorage(s.local_storage_dir)
    if name == "s3":
        return S3Storage(s)
    raise StorageError(f"Unknown storage backend {name!r}")
