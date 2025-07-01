import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load('best_model1.pkl')
scaler = joblib.load('scaler1.pkl')

# Judul
st.title("Prediksi Tingkat Stres Mahasiswa Selama Pembelajaran Daring")
st.write("Masukkan data berikut untuk memprediksi tingkat stres berdasarkan screen time dan tidur:")

# Input
screen_time = st.number_input("Screen time per hari (jam)", min_value=0.0, max_value=24.0, value=6.0)
sleep_duration = st.number_input("Tidur per hari (jam)", min_value=0.0, max_value=24.0, value=6.0)

if st.button("Prediksi"):
    X_input = np.array([[screen_time, sleep_duration]])
    X_scaled = scaler.transform(X_input)

    prediction = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    confidence = np.max(probabilities) * 100

    label_dict = {0: "Rendah", 1: "Sedang", 2: "Tinggi", 3: "Sangat Tinggi"}
    predicted_label = label_dict[prediction[0]]

    st.subheader("Hasil Prediksi:")
    st.success(f"Tingkat stres: **{predicted_label}**")
    st.write(f"Akurasi keyakinan model: **{confidence:.2f}%**")

    # Pesan sesuai level
    if prediction[0] == 3:
        st.error("🚨 Stres sangat tinggi. Cari bantuan profesional.")
    elif prediction[0] == 2:
        st.warning("⚠️ Stres tinggi. Jaga pola hidup sehat.")
    elif prediction[0] == 1:
        st.info("🔎 Stres sedang. Tetap seimbang.")
    else:
        st.success("✅ Stres rendah. Pertahankan gaya hidup ini.")
