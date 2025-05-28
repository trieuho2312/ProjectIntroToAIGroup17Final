import re

def clean_text(text):
    """
    Làm sạch lyrics:
    - Loại bỏ ký tự đặc biệt
    - Chuyển về chữ thường
    - Loại từ ngắn và từ rác phổ biến trong lời bài hát
    """
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))  # bỏ ký tự không phải chữ
    text = text.lower()

    # Các từ rác thường gặp trong lyrics
    trash_words = {
        "yeah", "uh", "la", "na", "woo", "oh", "ooh", "yo", "ay", "ha", "ah",
        "baby", "hey", "yea", "yah", "mmm", "huh", "whoa", "yuh", "ok", "okay"
    }

    words = [
        word for word in text.split()
        if len(word) > 2 and word not in trash_words
    ]

    return " ".join(words)

