import boto3
import pytest
from moto import mock_aws

from video_engine import config, storage
from video_engine.storage import LocalStorage, S3Storage, StorageError


def test_local_roundtrip(tmp_path):
    s = LocalStorage(tmp_path / "store")
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x" * 10)
    assert s.put_file(f, "jobs/1/input/a.mp4") == 10
    assert s.exists("jobs/1/input/a.mp4") and s.size("jobs/1/input/a.mp4") == 10
    assert s.url("jobs/1/input/a.mp4") == "/files/jobs/1/input/a.mp4"
    assert s.get_file("jobs/1/input/a.mp4", tmp_path / "ignored").read_bytes() == b"x" * 10
    s.delete_prefix("jobs/1")
    assert not s.exists("jobs/1/input/a.mp4")


def test_local_rejects_traversal(tmp_path):
    with pytest.raises(StorageError):
        LocalStorage(tmp_path).path("../../etc/passwd")


def test_flag_selects_backend(monkeypatch):
    assert storage.get_storage().name == "local"
    monkeypatch.setenv("VE_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("VE_S3_BUCKET", "")
    config.get_settings.cache_clear()
    storage.get_storage.cache_clear()
    with pytest.raises(StorageError, match="VE_S3_BUCKET"):
        storage.get_storage()


@mock_aws
def test_s3_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("VE_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("VE_S3_BUCKET", "videos")
    monkeypatch.setenv("VE_S3_REGION", "us-east-1")
    config.get_settings.cache_clear()
    storage.get_storage.cache_clear()
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="videos")

    s = storage.get_storage()
    assert isinstance(s, S3Storage)
    f = tmp_path / "out.mp4"
    f.write_bytes(b"video" * 100)
    assert s.put_file(f, "jobs/9/output/out.mp4", move=True) == 500
    assert not f.exists()
    assert s.exists("jobs/9/output/out.mp4") and s.size("jobs/9/output/out.mp4") == 500
    head = boto3.client("s3", region_name="us-east-1").head_object(Bucket="videos", Key="video-engine/jobs/9/output/out.mp4")
    assert head["ContentType"] == "video/mp4"
    url = s.url("jobs/9/output/out.mp4", download_name="out.mp4")
    assert "X-Amz-Signature" in url and "attachment" in url
    back = s.get_file("jobs/9/output/out.mp4", tmp_path / "dl.mp4")
    assert back.read_bytes() == b"video" * 100
    s.delete_prefix("jobs/9")
    assert not s.exists("jobs/9/output/out.mp4")
