import sys
import os
sys.path.append(os.path.abspath("."))

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from recommender.recommend import SongRecommender
from recommender.emotion_model import EmotionClassifier
from recommender.spotify_api import search_track
from recommender.genius_api import get_lyrics

st.set_page_config(page_title="🎵 Music Recommender", layout="wide")
st.title("🎵 Music Recommender System")
st.markdown("---")


def plot_similarity_chart(song_names, scores, save_path="similarity_chart.png"):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(song_names, scores, color='royalblue')
    plt.ylim(0, 1)
    plt.ylabel('Cosine Similarity')
    plt.title('Lyric Similarity with Input Song')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def display_song_result(row):
    col1, col2 = st.columns([1, 5])
    with col1:
        sp = search_track(f"{row['song']} {row['artist']}")
        if sp and sp['image']:
            st.image(sp['image'], width=100)
    with col2:
        st.markdown(f"**{row['song']}** - *{row['artist']}*")
        st.markdown(f"🎧 Emotion: **{row['emotion']}**")
        if sp and sp['url']:
            st.markdown(f"[Mở trên Spotify]({sp['url']})", unsafe_allow_html=True)


recommender = SongRecommender()
clf = EmotionClassifier()

option = st.radio(" Chọn chức năng:", (
    "Gợi ý bài hát tương tự",
    "Tìm bài hát theo lời nhập tự do"
), horizontal=False)

if option == "Gợi ý bài hát tương tự":
    song_name = st.text_input(" Nhập tên bài hát bạn đang nghe hoặc muốn gợi ý:")
    if song_name:
        df = recommender.df
        matched = df[df['song'].str.lower() == song_name.lower()]

        def show_result_list(title, results):
            if not results.empty:
                st.markdown(f"### {title}")
                for _, row in results.iterrows():
                    display_song_result(row)

        if not matched.empty:
            st.success(" Bài hát có trong hệ thống. Đang gợi ý bài tương tự...")
            original_emotion = matched.iloc[0]["emotion"]
            st.markdown(f" Cảm xúc gốc của bài hát: **{original_emotion.upper()}**")
            st.markdown("### 🎵 Bài hát bạn đã chọn:")
            display_song_result(matched.iloc[0])

            # Gợi ý
            filtered_results, other_results = recommender.recommend(song_name, max_total=10, max_same_emotion=5)

            total_shown = len(filtered_results) + len(other_results)
            st.markdown(f" Tổng số bài hát được gợi ý: `{total_shown}` (Cùng cảm xúc: {len(filtered_results)})")

            # Tính toán độ tương đồng
            vec = recommender.vectorizer.transform([matched.iloc[0]["cleaned_text"]])
            distances, indices = recommender.nn_model.kneighbors(vec, n_neighbors=6)
            similar_indices = indices[0][1:]
            recommended_df = recommender.df.iloc[similar_indices]
            similarity_scores = 1 - distances[0][1:]
            song_titles = recommended_df["song"].tolist()
            avg_similarity = similarity_scores.mean()
            match_count = sum(recommended_df["emotion"].str.lower() == original_emotion.lower())
            emotion_match_rate = match_count / len(recommended_df)

            # Hiển thị gợi ý
            show_result_list("Các bài hát tương tự có cùng cảm xúc:", filtered_results)
            show_result_list("Các bài hát tương tự nhưng khác cảm xúc:", other_results)

            # Biểu đồ
            st.markdown(f"** Cosine Similarity Trung bình:** `{avg_similarity:.2f}`")
            st.markdown(f"** Emotion Match Rate:** `{emotion_match_rate * 100:.1f}%`")
            plot_similarity_chart(song_titles, similarity_scores, "similarity_chart.png")
            st.image("similarity_chart.png", caption="Biểu đồ độ tương đồng lyrics", use_container_width=True)

            if total_shown == 0:
                st.warning(" Không tìm được bài hát tương tự trong hệ thống.")

        else:
            st.warning("Bài hát không có sẵn trong hệ thống. Đang tìm lyrics...")

            sp_result = search_track(song_name)
            if sp_result:
                st.markdown(f"**Đã tìm thấy:** `{sp_result['song']}` - *{sp_result['artist']}*")
                st.image(sp_result['image'], width=300)
                st.markdown(f"[Mở Spotify]({sp_result['url']})", unsafe_allow_html=True)

                lyrics = get_lyrics(sp_result['song'], sp_result['artist'])
                if lyrics:
                    emotion = clf.predict(lyrics)
                    st.markdown(f"🎧 Dự đoán cảm xúc bài hát: **{emotion.upper()}**")

                    filtered_results, other_results = recommender.recommend_from_lyrics(
                        lyrics, max_total=10, max_same_emotion=5
                    )
                    total_shown = len(filtered_results) + len(other_results)
                    st.markdown(f" Tổng số bài hát được gợi ý: `{total_shown}` (Cùng cảm xúc: {len(filtered_results)})")

                    # Cosine similarity
                    vec = recommender.vectorizer.transform([lyrics])
                    distances, indices = recommender.nn_model.kneighbors(vec, n_neighbors=5)
                    recommended_df = recommender.df.iloc[indices[0]]
                    similarity_scores = 1 - distances[0]
                    song_titles = recommended_df["song"].tolist()
                    avg_similarity = similarity_scores.mean()
                    emotion_match_rate = (len(filtered_results) / total_shown) if total_shown > 0 else 0

                    # Hiển thị kết quả
                    show_result_list("Các bài hát tương tự có cùng cảm xúc:", filtered_results)
                    show_result_list("Các bài hát tương tự nhưng khác cảm xúc:", other_results)

                    st.markdown(f"** Cosine Similarity Trung bình:** `{avg_similarity:.2f}`")
                    st.markdown(f"** Emotion Match Rate:** `{emotion_match_rate * 100:.1f}%`")
                    plot_similarity_chart(song_titles, similarity_scores, "similarity_chart.png")
                    st.image("similarity_chart.png", caption="Biểu đồ độ tương đồng lyrics", use_container_width=True)

                    if total_shown == 0:
                        st.warning("Không tìm được bài hát phù hợp với cảm xúc.")
                else:
                    st.error("Không tìm thấy lyrics để phân tích.")
            else:
                st.error("Không tìm thấy bài hát trên Spotify.")

elif option == "Tìm bài hát theo lời nhập tự do":
    user_input = st.text_input(" Nhập một câu mô tả cảm xúc, suy nghĩ hoặc lời bài hát:")
    if user_input:
        predicted_emotion = clf.predict(user_input)
        st.success(f" Dự đoán cảm xúc: **{predicted_emotion.upper()}**")

        df = recommender.df
        filtered = df[df["emotion"].str.lower() == predicted_emotion.lower()]
        if not filtered.empty:
            st.markdown("###  Các bài hát phù hợp với cảm xúc của bạn:")
            for _, row in filtered.sample(min(5, len(filtered))).iterrows():
                display_song_result(row)
        else:
            st.warning(" Không tìm thấy bài hát nào phù hợp với cảm xúc này.")
