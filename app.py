
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title='World Cup Predictor', layout='centered')

# Fungsi memuat model dengan error handling
@st.cache_resource
def load_my_model():
    try:
        return tf.keras.models.load_model('world_cup_winner_prediction_model.keras')
    except Exception as e:
        st.error(f'Gagal memuat model: {e}')
        return None

model = load_my_model()

st.title('🏆 World Cup Analysis Dashboard')

if model:
    with st.sidebar:
        st.header('Input Data Tim')
        team = st.text_input('Nama Tim', 'Argentina')
        continent = st.selectbox('Benua', ['Asia', 'Europe', 'South America', 'Africa', 'North America', 'Oceania'])
        wins = st.slider('Kemenangan (4thn)', 0, 50, 30)
        losses = st.slider('Kekalahan (4thn)', 0, 50, 5)
        rank = st.number_input('Peringkat FIFA', value=1)

    if st.button('Analisis'):
        # Menyesuaikan input dengan format training
        # Pastikan jumlah kolom sama dengan saat model dibuat
        input_dict = {
            'team': np.array([team]), 
            'continent': np.array([continent]),
            'wins_last_4y': np.array([wins], dtype=np.int64),
            'losses_last_4y': np.array([losses], dtype=np.int64),
            'goals_scored_last_4y': np.array([100], dtype=np.int64),
            'fifa_rank_pre_tournament': np.array([rank], dtype=np.int64),
            'version': np.array([2022], dtype=np.int64), 
            'is_host': np.array([0], dtype=np.int64),
            'goals_received_last_4y': np.array([30], dtype=np.int64), 
            'draws_last_4y': np.array([5], dtype=np.int64),
            'world_cup_titles_before': np.array([2], dtype=np.int64), 
            'squad_total_market_value_eur': np.array([6e8], dtype=np.float64),
            'fifa_points_pre_tournament': np.array([1700.0], dtype=np.float64), 
            'squad_avg_age': np.array([27.0], dtype=np.float64),
            'world_cup_participations_before': np.array([18], dtype=np.int64), 
            'groups_passed_before': np.array([10], dtype=np.int64),
            'round16_before': np.array([8], dtype=np.int64), 
            'quarterfinals_before': np.array([5], dtype=np.int64),
            'semifinals_before': np.array([4], dtype=np.int64), 
            'finals_before': np.array([3], dtype=np.int64)
        }

        pred = model.predict(input_dict)
        prob = float(pred[0][0])

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Lainnya', 'Peluang Menang'], [1-prob, prob], color=['#e0e0e0', '#2ecc71'])
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        st.write(f'### Probabilitas Menang: {prob:.2%}')
else:
    st.warning('Pastikan file world_cup_winner_prediction_model.keras sudah ada di GitHub.')
