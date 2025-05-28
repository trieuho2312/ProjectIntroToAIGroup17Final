import os
import json
import hashlib
import logging
from dotenv import load_dotenv
import lyricsgenius
from pathlib import Path

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load biến môi trường
load_dotenv(dotenv_path=Path(".env"))
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

if not GENIUS_ACCESS_TOKEN:
    raise ValueError("GENIUS_ACCESS_TOKEN chưa được thiết lập trong file .env")

# Khởi tạo Genius client
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=10, retries=3)
genius.skip_non_songs = True
genius.excluded_terms = ["(Remix)", "(Live)"]
genius.verbose = False

# Cấu hình thư mục cache
CACHE_DIR = Path("data/lyrics_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_cache_path(song_title, artist_name):
    key = f"{song_title.lower()}::{artist_name.lower()}"
    hashed = hashlib.md5(key.encode()).hexdigest()
    return CACHE_DIR / f"{hashed}.json"


def clean_lyrics(raw_lyrics: str) -> str:
    """Loại bỏ các đoạn không cần thiết trong lyrics như [Chorus], Embed, ..."""
    lines = raw_lyrics.splitlines()
    cleaned = [line for line in lines if not line.startswith("[") and "Embed" not in line]
    return "\n".join(cleaned).strip()


def get_lyrics(song_title: str, artist_name: str) -> str | None:
    """
    Trả về lời bài hát đã làm sạch từ Genius. Sử dụng cache nếu có.
    """
    cache_path = get_cache_path(song_title, artist_name)

    # Nếu đã có cache
    if cache_path.exists():
        logging.info(f" Đọc lyrics từ cache: {song_title} - {artist_name}")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("lyrics")

    # Nếu chưa có cache → gọi API
    try:
        logging.info(f" Gọi Genius API: {song_title} - {artist_name}")
        song = genius.search_song(title=song_title, artist=artist_name)
        if not song or not song.lyrics:
            logging.warning(" Không có lyrics hoặc không tìm thấy bài hát.")
            return None

        cleaned_lyrics = clean_lyrics(song.lyrics)

        # Lưu cache
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"lyrics": cleaned_lyrics}, f, ensure_ascii=False)

        return cleaned_lyrics

    except Exception as e:
        logging.error(f" Lỗi khi gọi Genius API: {e}")
        return None
