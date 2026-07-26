import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA World Cup Team Classification",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1a6b3c, #4CAF50, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a6b3c22, #4CAF5022);
        border: 1px solid #4CAF5055;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #4CAF50;
        border-left: 4px solid #FFD700;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">⚽ FIFA World Cup Team Classification</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning Dashboard — Prediksi Tim Juara Piala Dunia FIFA</p>', unsafe_allow_html=True)
st.divider()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/a/a9/FIFA_World_Cup_Official_Logo.png/200px-FIFA_World_Cup_Official_Logo.png", width=120)
    st.markdown("### ⚙️ Konfigurasi")
    uploaded_file = st.file_uploader("Upload file train.csv", type=["csv"])
    epochs = st.slider("Jumlah Epoch", min_value=5, max_value=50, value=20, step=5)
    test_size = st.slider("Ukuran Data Test (%)", min_value=10, max_value=30, value=20, step=5)
    run_button = st.button("🚀 Jalankan Model", use_container_width=True, type="primary")
    st.divider()
    st.markdown("**📌 Tentang App**")
    st.caption("Dashboard klasifikasi tim FIFA World Cup menggunakan Neural Network dengan TensorFlow/Keras.")

# ─── Helper: build & train model ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_and_train(df_raw, test_size_pct, n_epochs):
    leakage_cols = [c for c in ['finalist', 'semi_finalist', 'quarter_finalist'] if c in df_raw.columns]
    df = df_raw.drop(columns=leakage_cols)
    df.drop_duplicates(inplace=True)

    numeric_cols  = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    for col in numeric_cols:
        if col != 'winner':
            df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    train, test = train_test_split(df, test_size=test_size_pct/100,
                                   random_state=42, stratify=df['winner'])
    train, val  = train_test_split(train, test_size=0.2,
                                   random_state=42, stratify=train['winner'])

    def df_to_dataset(dataframe, shuffle=True, batch_size=32):
        dataframe = dataframe.copy()
        labels = dataframe.pop('winner')
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))
        return ds.batch(batch_size)

    train_ds = df_to_dataset(train)
    val_ds   = df_to_dataset(val,   shuffle=False)
    test_ds  = df_to_dataset(test,  shuffle=False)

    all_inputs, encoded_features = [], []
    for header in numeric_cols:
        if header == 'winner':
            continue
        inp = tf.keras.Input(shape=(1,), name=header)
        norm = tf.keras.layers.Normalization()
        norm.adapt(train[header].to_numpy().reshape(-1, 1))
        all_inputs.append(inp)
        encoded_features.append(norm(inp))

    for header in categorical_cols:
        inp = tf.keras.Input(shape=(1,), name=header, dtype='string')
        lookup = tf.keras.layers.StringLookup(vocabulary=train[header].unique())
        enc = tf.keras.layers.CategoryEncoding(
            num_tokens=lookup.vocabulary_size(), output_mode="binary")
        all_inputs.append(inp)
        encoded_features.append(enc(lookup(inp)))

    x = tf.keras.layers.concatenate(encoded_features)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(all_inputs, output)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    neg, pos = np.bincount(train['winner'])
    total = neg + pos
    class_weight = {0: (1/neg)*(total/2), 1: (1/pos)*(total/2)}

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=n_epochs, class_weight=class_weight, verbose=0)
    loss, accuracy = model.evaluate(test_ds, verbose=0)

    return df, train, val, test, model, history, loss, accuracy, numeric_cols, categorical_cols


# ─── Main Logic ──────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Silakan upload file **train.csv** di sidebar untuk memulai.", icon="📂")
    st.stop()

df_raw = pd.read_csv(uploaded_file)

if not run_button and 'results' not in st.session_state:
    st.info("File berhasil diupload. Klik **🚀 Jalankan Model** di sidebar untuk memproses.", icon="✅")
    st.dataframe(df_raw.head(10), use_container_width=True)
    st.stop()

if run_button or 'results' in st.session_state:
    if run_button:
        with st.spinner("⏳ Melatih model Neural Network... Harap tunggu."):
            results = build_and_train(df_raw, test_size, epochs)
        st.session_state['results'] = results

    (df, train, val, test, model, history,
     loss_val, accuracy_val, numeric_cols, categorical_cols) = st.session_state['results']

    # ── Top Metrics ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Total Data",    f"{len(df):,} baris")
    col2.metric("🏋️ Data Latih",   f"{len(train):,} baris")
    col3.metric("🎯 Akurasi Test",  f"{accuracy_val*100:.2f}%")
    col4.metric("📉 Loss Test",     f"{loss_val:.4f}")
    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🧠 Training", "🔍 Data Preview", "⚽ Prediksi"])

    # ── TAB 1: EDA ───────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        # Target distribution
        with c1:
            counts = df['winner'].value_counts().reset_index()
            counts.columns = ['Winner', 'Count']
            counts['Label'] = counts['Winner'].map({0: 'Tidak Menang', 1: 'Menang'})
            fig = px.pie(counts, values='Count', names='Label',
                         title='Distribusi Target (Winner)',
                         color_discrete_sequence=['#4CAF50','#FFD700'],
                         hole=0.4)
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        # Continent count
        with c2:
            if 'continent' in df.columns:
                cont_count = df['continent'].value_counts().reset_index()
                cont_count.columns = ['Continent', 'Count']
                fig2 = px.bar(cont_count, x='Count', y='Continent', orientation='h',
                              title='Jumlah Tim per Benua',
                              color='Count', color_continuous_scale='Greens')
                fig2.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig2, use_container_width=True)

        # Goals distribution
        c3, c4 = st.columns(2)
        if 'goals_scored_last_4y' in df.columns:
            with c3:
                fig3 = px.histogram(df, x='goals_scored_last_4y', nbins=30,
                                    title='Distribusi Goals Scored Last 4 Years',
                                    color_discrete_sequence=['#1a6b3c'],
                                    marginal='box')
                st.plotly_chart(fig3, use_container_width=True)

        if 'fifa_rank_pre_tournament' in df.columns:
            with c4:
                fig4 = px.histogram(df, x='fifa_rank_pre_tournament', nbins=30,
                                    title='Distribusi FIFA Rank Pre-Tournament',
                                    color_discrete_sequence=['#FFD700'],
                                    marginal='box')
                st.plotly_chart(fig4, use_container_width=True)

        # Correlation heatmap
        st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
        numerical_df = df.select_dtypes(include=['int64', 'float64'])
        corr = numerical_df.corr()
        fig5 = px.imshow(corr, text_auto='.2f', aspect='auto',
                         color_continuous_scale='RdYlGn',
                         title='Correlation Matrix — Fitur Numerik')
        fig5.update_layout(height=500)
        st.plotly_chart(fig5, use_container_width=True)

    # ── TAB 2: Training ───────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Hasil Training Neural Network</div>', unsafe_allow_html=True)

        hist = history.history
        epochs_range = list(range(1, len(hist['accuracy']) + 1))

        fig_train = make_subplots(rows=1, cols=2,
                                  subplot_titles=('Model Accuracy', 'Model Loss'))

        fig_train.add_trace(go.Scatter(x=epochs_range, y=hist['accuracy'],
                                       name='Train Accuracy', line=dict(color='#4CAF50', width=2)), row=1, col=1)
        fig_train.add_trace(go.Scatter(x=epochs_range, y=hist['val_accuracy'],
                                       name='Val Accuracy', line=dict(color='#FFD700', width=2, dash='dash')), row=1, col=1)
        fig_train.add_trace(go.Scatter(x=epochs_range, y=hist['loss'],
                                       name='Train Loss', line=dict(color='#1a6b3c', width=2)), row=1, col=2)
        fig_train.add_trace(go.Scatter(x=epochs_range, y=hist['val_loss'],
                                       name='Val Loss', line=dict(color='#FF6B35', width=2, dash='dash')), row=1, col=2)

        fig_train.update_xaxes(title_text="Epoch")
        fig_train.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig_train.update_yaxes(title_text="Loss", row=1, col=2)
        fig_train.update_layout(height=400, title_text="Kurva Training & Validasi")
        st.plotly_chart(fig_train, use_container_width=True)

        # Final epoch summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ Train Accuracy (final)", f"{hist['accuracy'][-1]*100:.2f}%")
        c2.metric("✅ Val Accuracy (final)",   f"{hist['val_accuracy'][-1]*100:.2f}%")
        c3.metric("📉 Train Loss (final)",     f"{hist['loss'][-1]:.4f}")
        c4.metric("📉 Val Loss (final)",        f"{hist['val_loss'][-1]:.4f}")

        # Model architecture summary
        st.markdown('<div class="section-header">Arsitektur Model</div>', unsafe_allow_html=True)
        arch_data = {
            "Layer": ["Input (Numeric)", "Normalization", "Input (Categorical)", "StringLookup + CategoryEncoding",
                      "Concatenate", "Dense (32, ReLU)", "Dropout (0.5)", "Dense (16, ReLU)", "Dense (1, Sigmoid)"],
            "Keterangan": ["Fitur numerik mentah", "Normalisasi otomatis per fitur",
                           "Fitur kategorikal (string)", "Encoding biner per kategori",
                           "Gabung semua fitur", "Hidden layer 1", "Regularisasi",
                           "Hidden layer 2", "Output probabilitas (klasifikasi biner)"]
        }
        st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

    # ── TAB 3: Data Preview ───────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Ringkasan Dataset</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Baris",   f"{df.shape[0]:,}")
        c2.metric("Kolom",   f"{df.shape[1]}")
        c3.metric("Missing (setelah cleaning)", "0")

        st.dataframe(df.head(20), use_container_width=True)

        st.markdown('<div class="section-header">Statistik Deskriptif</div>', unsafe_allow_html=True)
        st.dataframe(df.describe().T, use_container_width=True)

    # ── TAB 4: Prediksi ───────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">⚽ Prediksi Tim Baru</div>', unsafe_allow_html=True)
        st.caption("Isi nilai fitur di bawah untuk memprediksi apakah tim berpeluang menjadi juara.")

        input_data = {}
        cols = st.columns(3)
        num_features = [c for c in numeric_cols if c != 'winner']

        for i, col_name in enumerate(num_features):
            median_val = float(df[col_name].median())
            input_data[col_name] = cols[i % 3].number_input(
                col_name.replace('_', ' ').title(),
                value=median_val, format="%.2f", key=col_name)

        cat_cols_list = list(categorical_cols)
        for i, col_name in enumerate(cat_cols_list):
            options = sorted(df[col_name].unique().tolist())
            input_data[col_name] = cols[i % 3].selectbox(
                col_name.replace('_', ' ').title(), options=options, key=col_name)

        if st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True):
            input_df = pd.DataFrame([input_data])
            pred_ds = tf.data.Dataset.from_tensor_slices(dict(input_df)).batch(1)
            prob = float(model.predict(pred_ds, verbose=0)[0][0])

            st.divider()
            if prob >= 0.5:
                st.success(f"🏆 **Tim ini diprediksi MENANG!** (Probabilitas: {prob*100:.1f}%)")
            else:
                st.error(f"❌ **Tim ini diprediksi TIDAK Menang.** (Probabilitas menang: {prob*100:.1f}%)")

            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                title={'text': "Probabilitas Menang (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#4CAF50" if prob >= 0.5 else "#FF4444"},
                    'steps': [
                        {'range': [0, 50],  'color': "#ffdddd"},
                        {'range': [50, 100],'color': "#ddffdd"},
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'value': 50}
                }
            ))
            gauge.update_layout(height=300)
            st.plotly_chart(gauge, use_container_width=True)
