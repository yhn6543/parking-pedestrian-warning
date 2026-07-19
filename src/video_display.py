import shutil
from pathlib import Path


def get_file_size_mb(path: str | Path) -> float:
    file_path = Path(path)
    return file_path.stat().st_size / (1024 * 1024)


def read_video_bytes(video_path: str | Path) -> bytes:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file does not exist: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Video file is empty: {path}")
    with path.open("rb") as video_file:
        return video_file.read()


def get_ffmpeg_executable() -> str | None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
