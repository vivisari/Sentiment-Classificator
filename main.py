import streamlit as st
import pandas as pd
import torch
import os
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers.utils import logging

logging.set_verbosity_error()

# Saved Model Directory
model_drc = "saved_model/saved_models/cahya_distilbert-base-indonesian"

# Memastikan ada file "model_type"
config_path = os.path.join(model_drc, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r+", encoding="utf-8") as f:
        config_json = json.load(f)
        if "model_type" not in config_json:
            config_json["model_type"] = "distilbert"
            f.seek(0)
            json.dump(config_json, f, indent=2)
            f.truncate()
else:
    st.error("Missing config.json in model directory.")
    st.stop()

# Load Model DistillIndoBERT
try:
    config = AutoConfig.from_pretrained(model_drc)
    tokenizer = AutoTokenizer.from_pretrained(model_drc)
    model = AutoModelForSequenceClassification.from_pretrained(model_drc, config=config)
    num_labels = config.num_labels
except Exception as e:
    st.error(f"Error loading model/tokenizer: {e}")
    st.stop()

# Membuat class labels
class_names = ['Negatif', 'Positif']

st.title("Sentiment Analysis - DistilIndoBERT")
st.markdown("Pilih jenis input:")

input_type = st.radio("Pilih input:", ("Teks Langsung", "Unggah CSV"))

def predict_sentiment(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return probs, preds

# Option 1: Text input
if input_type == "Teks Langsung":
    text = st.text_area("Tulis ulasan layanan kesehatan di sini:", height=150)
    if st.button("Prediksi"):
        if text.strip() == "":
            st.warning("Harap masukkan teks ulasan.")
        else:
            probs, preds = predict_sentiment([text])
            predicted_class = preds.item()
            st.subheader("Hasil Prediksi:")
            st.write(f"**Sentimen:** {class_names[predicted_class]}")
            st.write(f"**Probabilitas:** {probs[0][predicted_class]:.4f}")

            st.subheader("Distribusi Probabilitas:")
            for i, prob in enumerate(probs[0]):
                st.write(f"{class_names[i]}: {prob:.4f}")

# Option 2: CSV upload
else:
    uploaded_file = st.file_uploader("Unggah file CSV (dengan kolom bernama 'review')", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "review" not in df.columns:
            st.error("File harus memiliki kolom bernama 'review'.")
        else:
            if st.button("Prediksi Semua"):
                texts = df["review"].astype(str).tolist()
                probs, preds = predict_sentiment(texts)

                df["Sentimen"] = [class_names[p] for p in preds.tolist()]
                for i in range(len(class_names)):
                    df[f"Prob_{class_names[i]}"] = probs[:, i].tolist()

                st.success("Prediksi selesai!")
                st.dataframe(df)

                # Donlod Hasil Prediksi (Opsional)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Unduh Hasil sebagai CSV", csv, "hasil_prediksi.csv", "text/csv")

                # Menampilkan distribusi data sentimen
                st.subheader("Ringkasan Distribusi Sentimen:")
                sentiment_counts = df["Sentimen"].value_counts()
                total = sentiment_counts.sum()

                for sentiment in class_names:
                    count = sentiment_counts.get(sentiment, 0)
                    percentage = (count / total) * 100 if total > 0 else 0
                    st.write(f"**{sentiment}:** {count} ({percentage:.2f}%)")