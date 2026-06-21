import runpod
import os
import base64
import json
import uuid
import logging
import subprocess
import binascii
import shutil
import boto3
from botocore.client import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CHECKPOINT_DIR   = os.getenv("CHECKPOINT_DIR", "/runpod-volume/weights/LongCat-Video-Avatar-1.5")
WORK_DIR         = os.getenv("WORK_DIR", "/LongCat-Video")
OUTPUT_BASE      = "/tmp/longcat_outputs"

R2_ACCOUNT_ID    = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_KEY    = os.getenv("R2_SECRET_KEY", "")
R2_BUCKET_NAME   = os.getenv("R2_BUCKET_NAME", "")
R2_PUBLIC_URL    = os.getenv("R2_PUBLIC_URL", "")

# ── Helpers ───────────────────────────────────────────────────────────────────

def truncate_b64(s, n=50):
    if not s:
        return "None"
    return f"{s[:n]}... ({len(s)} chars)" if len(s) > n else s


def upload_to_r2(file_path, object_key):
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    with open(file_path, "rb") as f:
        s3.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=object_key,
            Body=f,
            ContentType="video/mp4",
        )
    url = f"{R2_PUBLIC_URL.rstrip('/')}/{object_key}"
    logger.info(f"✅ R2 upload complete: {url}")
    return url


def save_base64(b64_data, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64_data))
        logger.info(f"✅ Saved base64 → {path}")
        return path
    except (binascii.Error, ValueError) as e:
        raise Exception(f"Base64 decode failed: {e}")


def download_url(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    result = subprocess.run(
        ["wget", "-O", path, "--no-verbose", "--timeout=30", url],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise Exception(f"wget failed: {result.stderr}")
    logger.info(f"✅ Downloaded {url} → {path}")
    return path


def resolve_input(job_input, key_path, key_url, key_b64, tmp_path):
    """Resolve a media input from path / url / base64."""
    if key_path in job_input:
        return job_input[key_path]
    elif key_url in job_input:
        return download_url(job_input[key_url], tmp_path)
    elif key_b64 in job_input:
        return save_base64(job_input[key_b64], tmp_path)
    return None


def find_output_video(output_dir):
    """Walk output_dir and return the first mp4 found."""
    for root, _, files in os.walk(output_dir):
        for f in files:
            if f.endswith(".mp4"):
                return os.path.join(root, f)
    return None


# ── Main handler ──────────────────────────────────────────────────────────────

def handler(job):
    job_input = job.get("input", {})

    # Log input (truncate base64 blobs)
    log_input = {
        k: truncate_b64(v) if "base64" in k else v
        for k, v in job_input.items()
    }
    logger.info(f"Job input: {log_input}")

    task_id  = f"task_{uuid.uuid4().hex[:8]}"
    tmp_dir  = f"/tmp/{task_id}"
    out_dir  = f"{OUTPUT_BASE}/{task_id}"
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # ── Mode detection ────────────────────────────────────────────────────────
    # person_count: "single" | "multi"
    # stage:        "ai2v" (image+audio→video) | "at2v" (text+audio→video)
    # resolution:   "480p" | "720p"
    person_count = job_input.get("person_count", "single")
    stage        = job_input.get("stage", "ai2v")          # ai2v or at2v
    resolution   = job_input.get("resolution", "480p")     # 480p or 720p
    num_segments = job_input.get("num_segments", 1)         # 1 = single clip, >1 = long video
    num_steps    = job_input.get("num_inference_steps", 8)  # 8 = distilled (fast), 50 = full
    ref_img_idx  = job_input.get("ref_img_index", 10)
    mask_range   = job_input.get("mask_frame_range", 3)
    prompt       = job_input.get("prompt", "A person talking naturally, realistic, high quality")
    audio_type   = job_input.get("audio_type", "para")      # para | add (multi only)

    logger.info(f"Mode: person_count={person_count}, stage={stage}, resolution={resolution}, num_segments={num_segments}")

    # ── Resolve image ─────────────────────────────────────────────────────────
    image_path = None
    if stage == "ai2v":
        image_path = resolve_input(
            job_input,
            "image_path", "image_url", "image_base64",
            f"{tmp_dir}/input_image.jpg"
        )
        if not image_path or not os.path.exists(image_path):
            return {"error": "stage=ai2v requires an image (image_path, image_url, or image_base64)"}

    # ── Resolve audio(s) ─────────────────────────────────────────────────────
    audio_path = resolve_input(
        job_input,
        "wav_path", "wav_url", "wav_base64",
        f"{tmp_dir}/audio_person1.wav"
    )
    if not audio_path or not os.path.exists(audio_path):
        return {"error": "Audio input required (wav_path, wav_url, or wav_base64)"}

    audio_path_2 = None
    if person_count == "multi":
        audio_path_2 = resolve_input(
            job_input,
            "wav_path_2", "wav_url_2", "wav_base64_2",
            f"{tmp_dir}/audio_person2.wav"
        )
        if not audio_path_2 or not os.path.exists(audio_path_2):
            return {"error": "Multi-person mode requires a second audio (wav_path_2, wav_url_2, or wav_base64_2)"}

    # ── Build input JSON for LongCat ─────────────────────────────────────────
    longcat_input = {"prompt": prompt}

    if stage == "ai2v":
        longcat_input["cond_image"] = image_path

    if person_count == "single":
        longcat_input["cond_audio"] = {"person1": audio_path}
    else:
        longcat_input["cond_audio"] = {
            "person1": audio_path,
            "person2": audio_path_2,
        }
        longcat_input["audio_type"] = audio_type
        # bbox is optional — if not provided, model uses default half-split
        if "bbox" in job_input:
            longcat_input["bbox"] = job_input["bbox"]

    input_json_path = f"{tmp_dir}/longcat_input.json"
    with open(input_json_path, "w") as f:
        json.dump(longcat_input, f, indent=2)
    logger.info(f"Input JSON written to {input_json_path}")

    # ── Build torchrun command ────────────────────────────────────────────────
    script = (
        "run_demo_avatar_single_audio_to_video.py"
        if person_count == "single"
        else "run_demo_avatar_multi_audio_to_video.py"
    )

    cmd = [
        "torchrun",
        "--nproc_per_node=1",           # single GPU on RunPod serverless
        script,
        f"--checkpoint_dir={CHECKPOINT_DIR}",
        f"--input_json={input_json_path}",
        f"--output_dir={out_dir}",
        f"--resolution={resolution}",
        f"--num_segments={num_segments}",
        f"--num_inference_steps={num_steps}",
        f"--ref_img_index={ref_img_idx}",
        f"--mask_frame_range={mask_range}",
        "--use_distill",
        "--model_type=avatar-v1.5",
        "--use_int8",
    ]

    # single only: add stage flag
    if person_count == "single":
        cmd.append(f"--stage_1={stage}")

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=WORK_DIR,
            capture_output=True,
            text=True,
            timeout=3600,   # 1hr max
        )
    except subprocess.TimeoutExpired:
        return {"error": "Inference timed out (>1 hour)"}

    if result.returncode != 0:
        logger.error(f"torchrun stderr:\n{result.stderr[-3000:]}")
        return {"error": f"Inference failed (exit {result.returncode})", "stderr": result.stderr[-1000:]}

    logger.info("✅ Inference complete")

    # ── Find output video ─────────────────────────────────────────────────────
    output_video = find_output_video(out_dir)
    if not output_video:
        return {"error": "No output video found after inference"}

    file_size = os.path.getsize(output_video)
    logger.info(f"Output video: {output_video} ({file_size} bytes)")

    # ── Upload to R2 → network volume → base64 fallback ──────────────────────
    if all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_KEY, R2_BUCKET_NAME, R2_PUBLIC_URL]):
        try:
            object_key = f"outputs/longcat_{task_id}.mp4"
            video_url = upload_to_r2(output_video, object_key)
            return {"video_url": video_url}
        except Exception as e:
            logger.error(f"❌ R2 upload failed: {e}")

    if job_input.get("network_volume", False):
        try:
            dest = f"/runpod-volume/outputs/longcat_{task_id}.mp4"
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(output_video, dest)
            logger.info(f"✅ Saved to network volume: {dest}")
            return {"video_path": dest}
        except Exception as e:
            logger.error(f"❌ Network volume save failed: {e}")

    # Last resort: base64
    try:
        with open(output_video, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
        logger.warning("⚠️  Returning base64 (R2 and volume both unavailable)")
        return {"video": video_b64}
    except Exception as e:
        return {"error": f"All output methods failed: {e}"}
    finally:
        # Cleanup temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
