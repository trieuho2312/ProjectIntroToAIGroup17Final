
# 🎵 Music Emotion-Based Recommender

Một hệ thống gợi ý bài hát dựa trên **nội dung lời bài hát** và **cảm xúc** sử dụng kỹ thuật **Xử lý ngôn ngữ tự nhiên (NLP)** và **Machine Learning**.

## 🚀 Tính năng chính

- ✅ Gợi ý bài hát tương tự dựa trên lời bài hát (TF-IDF + Cosine Similarity/Nearest Neighbor)
- ✅ Phân loại cảm xúc bài hát bằng mô hình Machine Learning (Logistic Regression )
- ✅ Tích hợp Spotify API (ảnh, liên kết nghe bài)
- ✅ Tự động phân tích cảm xúc từ lời nhập tự do hoặc từ bài hát mới
- ✅ Giao diện Streamlit thân thiện, trực quan
- ✅ Phân chia rõ bài gợi ý theo cảm xúc: **trùng mood** và **khác mood**

---

## 🗂️ Cấu trúc thư mục

```
.
├── app/                  # Giao diện Streamlit
│   └── app.py
├── recommender/          # Code xử lý logic AI
│   ├── recommend.py
│   ├── emotion_model.py
│   ├── spotify_api.py
│   ├── genius_api.py
│   └── ...
├── data/ spotify_dataset_10k.csv
├── models/               # Các file mô hình huấn luyện 
├── main.py               # File khởi động app
├── requirements.txt      # Danh sách thư viện
├── .env.example          # Mẫu biến môi trường
└── README.md
```

---

## ⚙️ Cách chạy ứng dụng

### 1. Clone project về

git clone https://github.com/your_username/your_repo.git
cd your_repo
```

### 2. Cài thư viện cần thiết
pip install -r requirements.txt
pip install -e.

### 3. Cấu hình biến môi trường

- Tạo file `.env` dựa theo `.env.example`
- Cung cấp các API key:

```env
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
GENIUS_ACCESS_TOKEN=your_genius_api_token
```

### 4. Chạy ứng dụng
b1 : xử lí dữ liệu đầu vào : python -m scripts.prepare_data
b2: huấn luyện mô hình phân tích bài hát tương tự : python -m scripts.build_embedding_model
b3: huấn luyện mô hình phân loại cảm xúc : python -m scripts.train_emotion_model
b4 : chạy chương trình: streamlit run app/app.py hoặc python app/main.py

➡️ Ứng dụng sẽ chạy tại `http://localhost:8501/`

---

## 🧠 Công nghệ sử dụng

- **Python**, **Streamlit** (giao diện web)
- **Scikit-learn** (TF-IDF, nearestneighbor, classification)
- **lyricsgenius**, **Spotipy** (API lấy dữ liệu)
- **dotenv**, **joblib**, **pandas**

---

## 📌 Ghi chú

- Dự án đã huấn luyện sẵn mô hình TF-IDF và cảm xúc, lưu tại `models/`
- Nếu bạn muốn huấn luyện lại, chạy script `build_embedding_model.py` và `train_emotion_model.py`
- Dự án hiện hoạt động tốt với 10.000+ bài hát tiếng Anh

---

## 📬 Liên hệ

HoTrieu-Hust Project