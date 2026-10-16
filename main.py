"""
Evana Media Suite — Backend API
- Video downloader via yt-dlp
- Image conversion via Pillow
- Video/Audio conversion via ffmpeg
"""
import os
import uuid
import shutil
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import yt_dlp
from PIL import Image
import imageio_ffmpeg

# ---------- Setup ----------
BASE_DIR = Path(__file__).parent
STORAGE = BASE_DIR / "storage"
DOWNLOADS = STORAGE / "downloads"
CONVERTS = STORAGE / "converts"
DOWNLOADS.mkdir(parents=True, exist_ok=True)
CONVERTS.mkdir(parents=True, exist_ok=True)

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

app = FastAPI(title="Evana API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ← في الإنتاج ضيّقها على نطاقك
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- In-memory job stores ----------
# (استبدلها بـ Redis / Firestore لو صار عندك ترافيك عالي)
DOWNLOAD_JOBS: dict = {}
CONVERT_JOBS: dict = {}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v"}
AUDIO_EXTS = {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"}


# ---------- Schemas ----------
class InfoRequest(BaseModel):
    url: str


class DownloadRequest(BaseModel):
    url: str
    format_id: Optional[str] = None
    audio_only: bool = False


# ---------- Health ----------
@app.get("/")
def health():
    return {"status": "ok", "service": "Evana API", "ffmpeg": os.path.basename(FFMPEG)}


# ==========================================================
#                  VIDEO DOWNLOADER
# ==========================================================
@app.post("/api/info")
async def get_info(req: InfoRequest):
    """جلب معلومات الفيديو (عنوان، صورة، صيغ متاحة) بدون تحميل."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(req.url, download=False)
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch info: {e}")

    formats = []
    seen = set()
    for f in info.get("formats", []):
        if f.get("vcodec") == "none" and f.get("acodec") == "none":
            continue
        key = (f.get("format_id"), f.get("ext"))
        if key in seen:
            continue
        seen.add(key)

        formats.append({
            "format_id": f.get("format_id"),
            "ext": f.get("ext"),
            "resolution": f.get("resolution") or f.get("format_note") or "audio",
            "filesize": f.get("filesize") or f.get("filesize_approx"),
            "vcodec": f.get("vcodec"),
            "acodec": f.get("acodec"),
            "abr": f.get("abr"),
            "is_audio_only": f.get("vcodec") == "none",
        })

    # نظّف الترتيب: أفضل جودة أولاً
    formats.sort(key=lambda x: (x["is_audio_only"], -(x["filesize"] or 0)))

    return {
        "title": info.get("title"),
        "thumbnail": info.get("thumbnail"),
        "duration": info.get("duration"),
        "uploader": info.get("uploader"),
        "webpage_url": info.get("webpage_url"),
        "formats": formats[:40],   # حد معقول للعرض
    }


def _run_download(job_id: str, url: str, format_id: Optional[str], audio_only: bool):
    """يعمل في الخلفية — يحمّل الفيديو ويحدّث الحالة."""
    job_dir = DOWNLOADS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    def hook(d):
        if d["status"] == "downloading":
            DOWNLOAD_JOBS[job_id]["progress"] = {
                "downloaded": d.get("downloaded_bytes", 0),
                "total": d.get("total_bytes") or d.get("total_bytes_estimate", 0),
                "speed": d.get("speed"),
                "eta": d.get("eta"),
                "percent": round(
                    (d.get("downloaded_bytes", 0) /
                     (d.get("total_bytes") or d.get("total_bytes_estimate") or 1)) * 100, 2
                ),
            }
        elif d["status"] == "finished":
            DOWNLOAD_JOBS[job_id]["progress"] = {"percent": 100, "finished": True}

    if audio_only:
        fmt = "bestaudio/best"
    elif format_id:
        fmt = f"{format_id}+bestaudio/{format_id}/best"
    else:
        fmt = "bestvideo+bestaudio/best"

    ydl_opts = {
        "outtmpl": str(job_dir / "%(title).120s.%(ext)s"),
        "format": fmt,
        "progress_hooks": [hook],
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "ffmpeg_location": FFMPEG,
        "merge_output_format": "mp4",
    }
    if audio_only:
        ydl_opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }]

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        files = [f for f in job_dir.glob("*") if f.is_file()]
        if not files:
            raise RuntimeError("No output file produced")

        output = max(files, key=lambda p: p.stat().st_size)  # أكبر ملف = الناتج النهائي

        DOWNLOAD_JOBS[job_id].update({
            "status": "completed",
            "file": str(output),
            "filename": output.name,
            "title": info.get("title"),
            "thumbnail": info.get("thumbnail"),
        })
    except Exception as e:
        DOWNLOAD_JOBS[job_id].update({"status": "failed", "error": str(e)})


@app.post("/api/download")
async def start_download(req: DownloadRequest, background: BackgroundTasks):
    job_id = uuid.uuid4().hex
    DOWNLOAD_JOBS[job_id] = {"status": "queued", "progress": {}, "url": req.url}
    background.add_task(_run_download, job_id, req.url, req.format_id, req.audio_only)
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    job = DOWNLOAD_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {
        "status": job["status"],
        "progress": job.get("progress"),
        "title": job.get("title"),
        "thumbnail": job.get("thumbnail"),
        "error": job.get("error"),
        "filename": job.get("filename"),
    }


@app.get("/api/file/{job_id}")
def get_downloaded_file(job_id: str):
    job = DOWNLOAD_JOBS.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(404, "File not ready")
    path = Path(job["file"])
    if not path.exists():
        raise HTTPException(404, "File missing")
    return FileResponse(path, filename=job["filename"], media_type="application/octet-stream")


# ==========================================================
#                  FORMAT CONVERTER
# ==========================================================
async def _ffmpeg_convert(input_path: Path, output_path: Path, target: str):
    """يستخدم ffmpeg لتحويل الفيديو/الصوت."""
    cmd = [FFMPEG, "-y", "-i", str(input_path)]

    if target in ("mp3", "wav", "aac", "flac", "ogg", "m4a"):
        # استخراج صوت
        cmd += ["-vn"]
        if target == "mp3":
            cmd += ["-acodec", "libmp3lame", "-b:a", "192k"]
        elif target == "m4a":
            cmd += ["-acodec", "aac", "-b:a", "192k"]
        else:
            cmd += ["-acodec", target]
    elif target in ("mp4", "mkv", "webm"):
        # تحويل فيديو
        if target == "webm":
            cmd += ["-c:v", "libvpx-vp9", "-c:a", "libopus", "-b:v", "2M"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-c", "copy"]

    cmd.append(str(output_path))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode()[-600:]}")


@app.post("/api/convert")
async def convert_file(
    file: UploadFile = File(...),
    target: str = Form(...),
):
    """
    target = الامتداد المطلوب بدون نقطة
             مثال: png / webp / jpg / mp4 / mp3
    """
    target = target.lower().strip().lstrip(".")
    job_id = uuid.uuid4().hex
    job_dir = CONVERTS / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / file.filename
    with input_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    ext = input_path.suffix.lower()
    output_path = job_dir / f"{input_path.stem}.{target}"

    try:
        # -------- IMAGE --------
        if ext in IMAGE_EXTS:
            img = Image.open(input_path)
            save_fmt = {"jpg": "JPEG", "jpeg": "JPEG"}.get(target, target.upper())
            if save_fmt == "JPEG":
                img = img.convert("RGB")
            elif save_fmt == "WEBP":
                img = img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else img.convert("RGB")
            img.save(output_path, format=save_fmt)

        # -------- VIDEO / AUDIO --------
        elif ext in VIDEO_EXTS or ext in AUDIO_EXTS:
            await _ffmpeg_convert(input_path, output_path, target)

        else:
            raise HTTPException(400, f"Unsupported input format: {ext}")

        CONVERT_JOBS[job_id] = {"path": str(output_path), "filename": output_path.name}

        # نظّف الملف الأصلي لتوفير مساحة
        try:
            input_path.unlink()
        except Exception:
            pass

        return {
            "job_id": job_id,
            "filename": output_path.name,
            "download_url": f"/api/convert/file/{job_id}",
        }
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(500, f"Conversion failed: {e}")


@app.get("/api/convert/file/{job_id}")
def get_converted_file(job_id: str):
    job = CONVERT_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    path = Path(job["path"])
    if not path.exists():
        raise HTTPException(404, "File missing")
    return FileResponse(path, filename=job["filename"], media_type="application/octet-stream")