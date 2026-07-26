import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
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
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #4CAF50;
        border-left: 4px solid #FFD700;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">⚽ FIFA World Cup Team Classification</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning Dashboard — Prediksi Tim Juara Piala Dunia FIFA</p>', unsafe_allow_html=True)
st.divider()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi")
    uploaded_file = st.file_uploader("Upload file train.csv", type=["csv"])
    test_size = st.slider("Ukuran Data Test (%)", min_value=10, max_value=30, value=20, step=5)
    hidden_layer = st.selectbox("Hidden Layer Size", ["(32, 16)", "(64, 32)", "(32, 16, 8)"], index=0)
    run_button = st.button("🚀 Jalankan Model", use_container_width=True, type="primary")
    st.divider()
    st.markdown("**📌 Tentang App**")
    st.caption("Dashboard klasifikasi tim FIFA World Cup menggunakan MLP Classifier (scikit-learn). Ringan dan cepat untuk Streamlit Cloud.")

# ─── Helper ──────────────────────────────────────────────────────────────────
def parse_hidden_layers(s):
    return tuple(int(x.strip()) for x in s.strip("()").split(","))

@st.cache_resource(show_spinner=False)
def build_and_train(df_raw, test_size_pct, hl_str):
    leakage_cols = [c for c in ['finalist', 'semi_finalist', 'quarter_finalist'] if c in df_raw.columns]
    df = df_raw.drop(columns=leakage_cols).copy()
    df.drop_duplicates(inplace=True)

    numeric_cols     = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in numeric_cols:
        if col != 'winner':
            df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = [c for c in df.columns if c != 'winner']
    X = df[feature_cols].values
    y = df['winner'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_pct/100, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    hl = parse_hidden_layers(hl_str)
    model = MLPClassifier(
        hidden_layer_sizes=hl,
        activation='relu',
        max_iter=1,
        warm_start=True,
        random_state=42,
        early_stopping=False,
    )

    train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []
    n_epochs = 50
    for epoch in range(n_epochs):
        model.fit(X_train_s, y_train)
        train_acc_list.append(accuracy_score(y_train, model.predict(X_train_s)))
        val_acc_list.append(accuracy_score(y_val, model.predict(X_val_s)))
        train_loss_list.append(log_loss(y_train, model.predict_proba(X_train_s)))
        val_loss_list.append(log_loss(y_val, model.predict_proba(X_val_s)))

    test_acc  = accuracy_score(y_test, model.predict(X_test_s))
    test_loss = log_loss(y_test, model.predict_proba(X_test_s))

    history = {
        'accuracy': train_acc_list, 'val_accuracy': val_acc_list,
        'loss': train_loss_list,    'val_loss': val_loss_list,
    }

    train_split = df.iloc[: len(X_train)]
    test_split  = df.iloc[-len(X_test):]

    return (df, train_split, X_val, X_test, y_test,
            model, scaler, encoders, feature_cols,
            history, test_loss, test_acc,
            numeric_cols, categorical_cols, df_raw)


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
        with st.spinner("⏳ Melatih model... Harap tunggu."):
            results = build_and_train(df_raw, test_size, hidden_layer)
        st.session_state['results'] = results

    (df, train_split, X_val, X_test, y_test,
     model, scaler, encoders, feature_cols,
     history, loss_val, accuracy_val,
     numeric_cols, categorical_cols, df_raw_orig) = st.session_state['results']

    # ── Top Metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Total Data",   f"{len(df):,} baris")
    c2.metric("🏋️ Data Latih",  f"{len(train_split):,} baris")
    c3.metric("🎯 Akurasi Test", f"{accuracy_val*100:.2f}%")
    c4.metric("📉 Loss Test",    f"{loss_val:.4f}")
    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🧠 Training", "🔍 Data Preview", "⚽ Prediksi"])

    # ── TAB 1: EDA ───────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            counts = df_raw_orig['winner'].value_counts().reset_index()
            counts.columns = ['Winner', 'Count']
            counts['Label'] = counts['Winner'].map({0: 'Tidak Menang', 1: 'Menang'})
            fig = px.pie(counts, values='Count', names='Label',
                         title='Distribusi Target (Winner)',
                         color_discrete_sequence=['#4CAF50','#FFD700'], hole=0.4)
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if 'continent' in df_raw_orig.columns:
                cont = df_raw_orig['continent'].value_counts().reset_index()
                cont.columns = ['Continent', 'Count']
                fig2 = px.bar(cont, x='Count', y='Continent', orientation='h',
                              title='Jumlah Tim per Benua',
                              color='Count', color_continuous_scale='Greens')
                fig2.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        if 'goals_scored_last_4y' in df_raw_orig.columns:
            with c3:
                fig3 = px.histogram(df_raw_orig, x='goals_scored_last_4y', nbins=30,
                                    title='Distribusi Goals Scored Last 4 Years',
                                    color_discrete_sequence=['#1a6b3c'], marginal='box')
                st.plotly_chart(fig3, use_container_width=True)
        if 'fifa_rank_pre_tournament' in df_raw_orig.columns:
            with c4:
                fig4 = px.histogram(df_raw_orig, x='fifa_rank_pre_tournament', nbins=30,
                                    title='Distribusi FIFA Rank Pre-Tournament',
                                    color_discrete_sequence=['#FFD700'], marginal='box')
                st.plotly_chart(fig4, use_container_width=True)

        st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
        num_df = df_raw_orig.select_dtypes(include=['int64','float64'])
        fig5 = px.imshow(num_df.corr(), text_auto='.2f', aspect='auto',
                         color_continuous_scale='RdYlGn',
                         title='Correlation Matrix — Fitur Numerik')
        fig5.update_layout(height=500)
        st.plotly_chart(fig5, use_container_width=True)

    # ── TAB 2: Training ───────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Kurva Training Model (MLP)</div>', unsafe_allow_html=True)

        epochs_range = list(range(1, len(history['accuracy']) + 1))
        fig_train = make_subplots(rows=1, cols=2, subplot_titles=('Akurasi per Epoch', 'Loss per Epoch'))
        fig_train.add_trace(go.Scatter(x=epochs_range, y=history['accuracy'],
                                       name='Train Acc', line=dict(color='#4CAF50', width=2)), row=1, col=1)
        fig_train.add_trace(go.Scatter(x=epochs_range, y=history['val_accuracy'],
                                       name='Val Acc', line=dict(color='#FFD700', width=2, dash='dash')), row=1, col=1)
        fig_train.add_trace(go.Scatter(x=epochs_range, y=history['loss'],
                                       name='Train Loss', line=dict(color='#1a6b3c', width=2)), row=1, col=2)
        fig_train.add_trace(go.Scatter(x=epochs_range, y=history['val_loss'],
                                       name='Val Loss', line=dict(color='#FF6B35', width=2, dash='dash')), row=1, col=2)
        fig_train.update_xaxes(title_text="Epoch")
        fig_train.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig_train.update_yaxes(title_text="Loss",     row=1, col=2)
        fig_train.update_layout(height=400)
        st.plotly_chart(fig_train, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("✅ Train Acc (akhir)",  f"{history['accuracy'][-1]*100:.2f}%")
        c2.metric("✅ Val Acc (akhir)",    f"{history['val_accuracy'][-1]*100:.2f}%")
        c3.metric("📉 Train Loss (akhir)", f"{history['loss'][-1]:.4f}")
        c4.metric("📉 Val Loss (akhir)",   f"{history['val_loss'][-1]:.4f}")

        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        y_pred = model.predict(scaler.transform(X_test))
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, aspect='auto',
                           labels=dict(x="Prediksi", y="Aktual"),
                           x=['Tidak Menang','Menang'], y=['Tidak Menang','Menang'],
                           color_continuous_scale='Greens',
                           title='Confusion Matrix — Data Test')
        fig_cm.update_layout(height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown('<div class="section-header">Arsitektur Model</div>', unsafe_allow_html=True)
        arch = {
            "Komponen": ["Algoritma","Hidden Layers","Aktivasi","Optimizer","Input Features","Output"],
            "Detail":   ["MLP Classifier (scikit-learn)",
                         str(parse_hidden_layers(hidden_layer)),
                         "ReLU","Adam (default)",
                         f"{len(feature_cols)} fitur",
                         "Klasifikasi biner (0/1)"]
        }
        st.dataframe(pd.DataFrame(arch), use_container_width=True, hide_index=True)

    # ── TAB 3: Data Preview ───────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Ringkasan Dataset</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Baris",  f"{df_raw_orig.shape[0]:,}")
        c2.metric("Kolom",  f"{df_raw_orig.shape[1]}")
        c3.metric("Missing (setelah cleaning)", "0")
        st.dataframe(df_raw_orig.head(20), use_container_width=True)
        st.markdown('<div class="section-header">Statistik Deskriptif</div>', unsafe_allow_html=True)
        st.dataframe(df_raw_orig.describe().T, use_container_width=True)

    # ── TAB 4: Prediksi ───────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">⚽ Prediksi Tim Baru</div>', unsafe_allow_html=True)
        st.caption("Isi nilai fitur di bawah untuk memprediksi apakah tim berpeluang menjadi juara.")

        input_data = {}
        form_cols = st.columns(3)
        num_features = [c for c in numeric_cols if c != 'winner']

        for i, col_name in enumerate(num_features):
            median_val = float(df_raw_orig[col_name].median())
            input_data[col_name] = form_cols[i % 3].number_input(
                col_name.replace('_', ' ').title(),
                value=median_val, format="%.2f", key=f"pred_{col_name}")

        for i, col_name in enumerate(categorical_cols):
            options = sorted(df_raw_orig[col_name].dropna().unique().tolist())
            input_data[col_name] = form_cols[i % 3].selectbox(
                col_name.replace('_', ' ').title(), options=options, key=f"pred_{col_name}")

        if st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True):
            input_row = []
            for col_name in feature_cols:
                val = input_data[col_name]
                if col_name in encoders:
                    val = encoders[col_name].transform([str(val)])[0]
                input_row.append(float(val))

            input_scaled = scaler.transform([input_row])
            prob = model.predict_proba(input_scaled)[0][1]

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
                        {'range': [0, 50],   'color': "#ffdddd"},
                        {'range': [50, 100], 'color': "#ddffdd"},
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'value': 50}
                }
            ))
            gauge.update_layout(height=300)
            st.plotly_chart(gauge, use_container_width=True)
