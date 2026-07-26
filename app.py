
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Konfigurasi Halaman
st.set_page_config(page_title='World Cup Predictor', layout='centered')

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('world_cup_winner_prediction_model.keras')

model = load_my_model()

st.title('🏆 Prediksi Peluang Juara Dunia')
st.write('Masukkan data tim untuk melihat analisis visual peluang kemenangan.')

# Sidebar untuk Input
with st.sidebar:
    st.header('Parameter Tim')
    team = st.text_input('Nama Tim', 'Indonesia')
    continent = st.selectbox('Benua', ['Asia', 'Europe', 'South America', 'Africa', 'North America', 'Oceania'])
    wins = st.slider('Kemenangan (4 thn terakhir)', 0, 50, 20)
    losses = st.slider('Kekalahan (4 thn terakhir)', 0, 50, 10)
    rank = st.number_input('Peringkat FIFA', value=100)

if st.button('Analisis Peluang'):
    # Dummy data untuk fitur lainnya (sesuaikan dengan input model Anda)
    input_dict = {
        'team': np.array([team]), 'continent': np.array([continent]),
        'wins_last_4y': np.array([wins], dtype=np.int64),
        'losses_last_4y': np.array([losses], dtype=np.int64),
        'goals_scored_last_4y': np.array([60], dtype=np.int64),
        'fifa_rank_pre_tournament': np.array([rank], dtype=np.int64),
        'version': np.array([2022], dtype=np.int64), 'is_host': np.array([0], dtype=np.int64),
        'goals_received_last_4y': np.array([40], dtype=np.int64), 'draws_last_4y': np.array([10], dtype=np.int64),
        'world_cup_titles_before': np.array([0], dtype=np.int64), 'squad_total_market_value_eur': np.array([1e8], dtype=np.float64),
        'fifa_points_pre_tournament': np.array([1000.0], dtype=np.float64), 'squad_avg_age': np.array([26.0], dtype=np.float64),
        'world_cup_participations_before': np.array([0], dtype=np.int64), 'groups_passed_before': np.array([0], dtype=np.int64),
        'round16_before': np.array([0], dtype=np.int64), 'quarterfinals_before': np.array([0], dtype=np.int64),
        'semifinals_before': np.array([0], dtype=np.int64), 'finals_before': np.array([0], dtype=np.int64)
    }

    prediction = model.predict(input_dict)
    prob_win = float(prediction[0][0])
    prob_lose = 1 - prob_win

    # Visualisasi Utama
    st.subheader(f'Hasil Analisis: {team}')
    
    fig, ax = plt.subplots()
    colors = ['#FF4B4B', '#1C83E1']
    ax.bar(['Peluang Kalah', 'Peluang Menang'], [prob_lose, prob_win], color=colors)
    ax.set_ylim(0, 1)
    for i, v in enumerate([prob_lose, prob_win]):
        ax.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
    
    st.pyplot(fig)
    
    if prob_win > 0.5:
        st.success(f'Tim {team} memiliki peluang menang yang kuat!')
    else:
        st.warning(f'Tim {team} perlu usaha ekstra untuk menang.')
