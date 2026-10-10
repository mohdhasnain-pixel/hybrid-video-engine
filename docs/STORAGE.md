# Storage: local disk or S3

<p align="center"><img src="images/storage-flag.svg" alt="Storage feature flag" width="100%"></p>

One environment variable, `VE_STORAGE_BACKEND`, decides where **uploads, enhanced outputs and comparison assets**
are kept.

| Value | Where files live | How the browser gets them |
|---|---|---|
| `local` *(default)* | `VE_LOCAL_STORAGE_DIR` (default `./data/storage`) on this computer | Served by the app at `/files/…` with HTTP range support, so videos can seek |
| `s3` | `s3://$VE_S3_BUCKET/$VE_S3_PREFIX` | SigV4 presigned URLs (streaming, plus download URLs with `Content-Disposition`) |

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `VE_STORAGE_BACKEND` | `local` | `local` or `s3` |
| `VE_LOCAL_STORAGE_DIR` | `./data/storage` | Root for the local backend |
| `VE_S3_BUCKET` | — | Required for `s3` |
| `VE_S3_PREFIX` | `video-engine/` | Key prefix inside the bucket |
| `VE_S3_REGION` | — | e.g. `us-east-1` |
| `VE_S3_ENDPOINT_URL` | — | For MinIO or LocalStack |
| `VE_S3_PRESIGN_EXPIRY_SECONDS` | `3600` | Lifetime of the browser URLs |

AWS credentials come from the standard chain: `AWS_PROFILE`, `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, SSO, or an
instance role. Copy [`.env.example`](../.env.example) to `.env` to set everything in one place.

## Behaviour

- **Each job records its backend.** Jobs created before you flip the flag stay viewable as long as their storage
  still exists.
- **S3 mode still needs some local disk while a job runs.** FFmpeg and the models need real files, so the upload is
  kept in `VE_WORK_DIR` for processing, uploaded to S3, and the work directory is deleted when the job ends.
- **Local mode processes the upload in place:** no extra copy.
- **Deleting a job** deletes its files from whichever backend holds them.
- **Job metadata** (small JSON files) always lives in `VE_JOBS_DIR` (`./data/jobs`).

Key layout (identical in both backends):

```
jobs/<job-id>/input/<original-filename>
jobs/<job-id>/output/<name>_enhanced.mp4
jobs/<job-id>/compare/{source_preview,output_preview}.mp4, poster_*.jpg, crop_*.png
```

## Testing S3 without AWS

The test suite uses [moto](https://github.com/getmoto/moto) to run a full job end to end in S3 mode (upload →
process → presigned URLs → delete) with no AWS account. For a real local S3, start MinIO and set
`VE_S3_ENDPOINT_URL=http://localhost:9000`.
