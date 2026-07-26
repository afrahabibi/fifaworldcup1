import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title='World Cup Predictor')

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('world_cup_winner_prediction_model.keras')
    except Exception as e:
        st.error(f'Model tidak ditemukan atau rusak: {e}')
        return None

model = load_model()

st.title('🏆 Analisis Peluang Juara')

if model:
    team = st.text_input('Nama Tim', 'Brazil')
    continent = st.selectbox('Benua', ['South America', 'Europe', 'North America', 'Asia', 'Africa', 'Oceania'])
    
    if st.button('Analisis'):
        try:
            # Menyiapkan dummy input untuk seluruh 20 fitur model
            dummy_input = {}
            for name in model.input_names:
                if name == 'team':
                    dummy_input[name] = np.array([[team]])
                elif name == 'continent':
                    dummy_input[name] = np.array([[continent]])
                else:
                    # Nilai numerik default untuk fitur lainnya
                    dummy_input[name] = np.array([[0.0]], dtype=np.float32)
            
            pred = model.predict(dummy_input)
            prob = float(pred[0][0])
            
            fig, ax = plt.subplots()
            ax.pie([1-prob, prob], labels=['Lainnya', 'Peluang'], autopct='%1.1f%%', colors=['#eeeeee', '#2ecc71'])
            st.pyplot(fig)
        except Exception as e:
            st.error(f'Error Prediksi: {e}')
else:
    st.info('Silakan unggah file model .keras ke GitHub Anda.')
