# longcat-avatar-worker

RunPod serverless worker for [LongCat-Video-Avatar 1.5](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5) — audio-driven avatar video generation.

## Setup

### 1. Network Volume (weights)

Create a RunPod Network Volume (60GB+) and download weights onto it once using an on-demand pod:

```bash
pip install "huggingface_hub[hf_transfer,cli]"
export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli download meituan-longcat/LongCat-Video \
    --local-dir /runpod-volume/weights/LongCat-Video

huggingface-cli download meituan-longcat/LongCat-Video-Avatar-1.5 \
    --local-dir /runpod-volume/weights/LongCat-Video-Avatar-1.5
```

### 2. Docker Hub secrets

Add to GitHub repo → Settings → Secrets:
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

### 3. RunPod endpoint

- GPU: A100 80GB
- Image: `finalta/longcat-avatar-worker:latest`
- Volume mount: `/runpod-volume`
- Container disk: 20GB
- Workers Min: 0 (scale to zero)
- FlashBoot: enabled
- Env vars:
  - `CHECKPOINT_DIR=/runpod-volume/weights/LongCat-Video-Avatar-1.5`
  - `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_KEY`, `R2_BUCKET_NAME`, `R2_PUBLIC_URL`

---

## API

### Input parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `person_count` | string | `"single"` | `"single"` or `"multi"` |
| `stage` | string | `"ai2v"` | `"ai2v"` (image+audio) or `"at2v"` (text+audio, no image) |
| `resolution` | string | `"480p"` | `"480p"` (480×832) or `"720p"` (768×1280) |
| `num_segments` | int | `1` | Number of video segments. `1` = single clip (~5s). `5` = ~25s long video |
| `num_inference_steps` | int | `8` | `8` = distilled fast. `50` = full quality (slow) |
| `prompt` | string | - | Text description of the scene |
| `image_url` / `image_base64` / `image_path` | string | - | Reference portrait (required for `ai2v`) |
| `wav_url` / `wav_base64` / `wav_path` | string | - | Audio for person 1 |
| `wav_url_2` / `wav_base64_2` / `wav_path_2` | string | - | Audio for person 2 (multi only) |
| `audio_type` | string | `"para"` | `"para"` = simultaneous, `"add"` = sequential (multi only) |
| `bbox` | object | auto | Bounding boxes for multi-person: `{"person1": [y_min, x_min, y_max, x_max], "person2": [...]}` |
| `ref_img_index` | int | `10` | Reference frame index for long video continuation |
| `mask_frame_range` | int | `3` | Overlap frames between segments |
| `network_volume` | bool | `false` | Save output to `/runpod-volume/outputs/` instead of R2 |

### Output

```json
{ "video_url": "https://your-r2-domain/outputs/longcat_abc123.mp4" }
```

Falls back to `{ "video_path": "/runpod-volume/outputs/..." }` or `{ "video": "<base64>" }` if R2 is not configured.

### Example payloads

See `examples/` folder.
