import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Risk Explorer",
    page_icon="💗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Pastel, cheerful styling
# ----------------------------------------------------------------------------
PASTEL_CSS = """
<style>
:root{
    --mint:#CFF3E1;
    --lavender:#E5DEFF;
    --peach:#FFE2D1;
    --sky:#D6EEFF;
    --butter:#FFF6C9;
    --pink:#FFD9EC;
    --ink:#4A4458;
    --ink-soft:#6E6680;
}

html, body, [class*="css"]  {
    font-family: "Poppins", "Segoe UI", sans-serif;
    color: var(--ink);
}

.stApp {
    background: linear-gradient(160deg, #FDF9FF 0%, #F4FBFF 45%, #FFFDF6 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--lavender) 0%, var(--sky) 55%, var(--mint) 100%);
    border-right: none;
}
section[data-testid="stSidebar"] * {
    color: var(--ink) !important;
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2{
    font-weight: 700;
}

/* Headings */
h1, h2, h3 {
    color: var(--ink);
    font-weight: 700;
}
h1 { letter-spacing: 0.5px; }

/* Cards */
.pastel-card {
    background: #FFFFFFB0;
    border-radius: 18px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 18px rgba(150, 130, 200, 0.12);
    border: 1px solid #ffffff90;
}

.metric-pill {
    display:inline-block;
    padding: 0.55rem 1.1rem;
    border-radius: 999px;
    font-weight:600;
    margin: 0.25rem 0.4rem 0.25rem 0;
    font-size: 0.95rem;
}
.pill-mint { background: var(--mint); color:#1f6b4f;}
.pill-lavender { background: var(--lavender); color:#5a4b9c;}
.pill-peach { background: var(--peach); color:#9c5a2e;}
.pill-sky { background: var(--sky); color:#2e6f9c;}
.pill-pink { background: var(--pink); color:#a13d72;}
.pill-butter { background: var(--butter); color:#8a7400;}

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(135deg, #FFD1E8, #D6EEFF);
    color: #5a4b9c;
    border: none;
    border-radius: 14px;
    padding: 0.6rem 1.4rem;
    font-weight: 700;
    box-shadow: 0 3px 10px rgba(150,130,200,0.25);
}
.stButton>button:hover {
    background: linear-gradient(135deg, #D6EEFF, #CFF3E1);
    color: #2e6f9c;
}

/* Radio (nav) */
div[role="radiogroup"] label {
    background: #FFFFFFAA;
    border-radius: 12px;
    padding: 0.45rem 0.8rem;
    margin-bottom: 0.35rem;
    display:block;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

hr {
    border: none;
    border-top: 2px dashed #E5DEFF;
    margin: 1.2rem 0;
}
</style>
"""
st.markdown(PASTEL_CSS, unsafe_allow_html=True)

PASTEL_PALETTE = ["#A7E3C5", "#B7B0F2", "#FFC8A2", "#9FD3F0", "#F7B6D2", "#FFE08A"]
sns.set_palette(PASTEL_PALETTE)
plt.rcParams["axes.facecolor"] = "#FFFFFF00"
plt.rcParams["figure.facecolor"] = "#FFFFFF00"
plt.rcParams["axes.edgecolor"] = "#B6AEDB"
plt.rcParams["axes.labelcolor"] = "#4A4458"
plt.rcParams["xtick.color"] = "#4A4458"
plt.rcParams["ytick.color"] = "#4A4458"
plt.rcParams["text.color"] = "#4A4458"

TARGET = "Heart_Risk"

FEATURE_INFO = {
    "Chest_Pain": "Chest pain or tightness",
    "Shortness_of_Breath": "Shortness of breath",
    "Fatigue": "Unusual fatigue / tiredness",
    "Palpitations": "Heart palpitations (racing/irregular heartbeat)",
    "Dizziness": "Dizziness or light-headedness",
    "Swelling": "Swelling in legs, ankles, or feet",
    "Pain_Arms_Jaw_Back": "Pain radiating to arms, jaw, or back",
    "Cold_Sweats_Nausea": "Cold sweats or nausea",
    "High_BP": "Diagnosed high blood pressure",
    "High_Cholesterol": "Diagnosed high cholesterol",
    "Diabetes": "Diagnosed diabetes",
    "Smoking": "Currently smokes",
    "Obesity": "Clinically classified as obese",
    "Sedentary_Lifestyle": "Mostly sedentary lifestyle (little exercise)",
    "Family_History": "Family history of heart disease",
    "Chronic_Stress": "Chronic / ongoing stress",
}
SYMPTOM_COLS = ["Chest_Pain", "Shortness_of_Breath", "Fatigue", "Palpitations",
                 "Dizziness", "Swelling", "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea"]
RISK_FACTOR_COLS = ["High_BP", "High_Cholesterol", "Diabetes", "Smoking", "Obesity",
                     "Sedentary_Lifestyle", "Family_History", "Chronic_Stress"]
ALL_BINARY_COLS = SYMPTOM_COLS + RISK_FACTOR_COLS + ["Gender"]


# ----------------------------------------------------------------------------
# Data + model pipeline (cached so it only runs once)
# ----------------------------------------------------------------------------
@st.cache_data
def load_raw_data():
    df = pd.read_csv("heart_disease.csv")
    n_before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df_clean)
    return df, df_clean, n_before, n_after


@st.cache_resource
def build_pipeline(df_clean):
    feature_cols = [c for c in df_clean.columns if c != TARGET]
    X = df_clean[feature_cols]
    y = df_clean[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train_scaled, y_train)

    results = {}
    for name, model in [("Logistic Regression", log_reg), ("Random Forest", rf)]:
        pred = model.predict(X_test_scaled)
        results[name] = {
            "model": model,
            "pred": pred,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "cm": confusion_matrix(y_test, pred),
        }

    return {
        "feature_cols": feature_cols,
        "scaler": scaler,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "results": results,
    }


df_raw, df, n_before, n_after = load_raw_data()
pipe = build_pipeline(df)


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.markdown("## 💗 Heart Risk Explorer")
st.sidebar.markdown("Walk through the whole project, step by step.")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to:",
    [
        "1 · Data Overview",
        "2 · EDA",
        "3 · Cleaning & Preprocessing",
        "4 · Model Results",
        "5 · Try a Live Prediction",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<span class='metric-pill pill-mint'>63,755 rows</span>"
    "<span class='metric-pill pill-lavender'>18 features</span>"
    "<span class='metric-pill pill-peach'>2 models</span>",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# 1. Data Overview
# ----------------------------------------------------------------------------
if page == "1 · Data Overview":
    st.title("💗 Heart Disease Risk — Data Overview")

    st.markdown(
        """
        <div class="pastel-card">
        This app walks you through a complete machine learning project that predicts a person's
        <b>heart disease risk</b> from their reported symptoms, lifestyle factors, and medical history.
        The dataset is <b>synthetic</b>: each row represents one patient, with binary (yes/no) indicators
        for symptoms and risk factors, an age value, and a computed risk label.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='pastel-card' style='text-align:center'><h3>{df.shape[0]:,}</h3>Patients (after cleaning)</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='pastel-card' style='text-align:center'><h3>{df.shape[1]}</h3>Columns</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='pastel-card' style='text-align:center'><h3>{(df[TARGET]==1).mean()*100:.1f}%</h3>Labeled at-risk</div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='pastel-card' style='text-align:center'><h3>{int(df['Age'].min())}–{int(df['Age'].max())}</h3>Age range</div>", unsafe_allow_html=True)

    st.markdown("### A peek at the data")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### What each column means")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Symptoms**")
        for c in SYMPTOM_COLS:
            st.markdown(f"- `{c}` — {FEATURE_INFO[c]}")
    with col2:
        st.markdown("**Risk factors & demographics**")
        for c in RISK_FACTOR_COLS:
            st.markdown(f"- `{c}` — {FEATURE_INFO[c]}")
        st.markdown("- `Gender` — 0 = Female, 1 = Male")
        st.markdown("- `Age` — patient age in years")
        st.markdown(f"- `{TARGET}` — **target**: 0 = No Risk, 1 = At Risk")


# ----------------------------------------------------------------------------
# 2. EDA
# ----------------------------------------------------------------------------
elif page == "2 · EDA":
    st.title("🔎 Exploratory Data Analysis")

    st.markdown(
        "<div class='pastel-card'>A quick visual tour of the dataset — what the features look like, "
        "and how they relate to heart risk.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Age distribution by heart risk")
    fig, ax = plt.subplots(figsize=(7, 4))
    for risk, color, label in [(0, "#A7E3C5", "No Risk"), (1, "#F7B6D2", "At Risk")]:
        ax.hist(df[df[TARGET] == risk]["Age"], bins=25, alpha=0.75, color=color, label=label, edgecolor="white")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)
    st.caption("Older patients show a noticeably higher share of 'At Risk' cases.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Heart risk % by age group")
        age_groups = pd.cut(df["Age"], bins=[20, 35, 50, 65, 85], labels=["20–35", "35–50", "50–65", "65+"])
        age_risk = df.groupby(age_groups, observed=True)[TARGET].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 4))
        age_risk.plot(kind="bar", color=PASTEL_PALETTE, edgecolor="white", ax=ax)
        ax.set_ylabel("% At Risk")
        ax.set_xlabel("Age group")
        plt.xticks(rotation=0)
        st.pyplot(fig)

    with col2:
        st.markdown("### Heart risk % by gender")
        gender_risk = df.groupby("Gender")[TARGET].mean() * 100
        fig, ax = plt.subplots(figsize=(5, 4))
        gender_risk.plot(kind="bar", color=["#F7B6D2", "#9FD3F0"], edgecolor="white", ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Female", "Male"], rotation=0)
        ax.set_ylabel("% At Risk")
        st.pyplot(fig)

    st.markdown("### How strongly each feature relates to heart risk")
    corr = df.select_dtypes(include=["int64", "float64"]).corr()
    feat_corr = corr[TARGET].drop(TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#F7B6D2" if v > 0 else "#9FD3F0" for v in feat_corr.values]
    ax.barh(feat_corr.index, feat_corr.values, color=colors, edgecolor="white")
    ax.axvline(0, color="#B6AEDB", linewidth=1, linestyle="--")
    ax.set_xlabel("Correlation with Heart_Risk")
    st.pyplot(fig)
    st.caption("Symptoms like chest pain and pain in arms/jaw/back, plus risk factors like high BP and smoking, "
               "correlate most strongly with heart risk.")

    st.markdown("### Full correlation heatmap")
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr, cmap="PuBu", annot=True, fmt=".2f", linewidths=0.4, annot_kws={"size": 6}, ax=ax)
    st.pyplot(fig)


# ----------------------------------------------------------------------------
# 3. Cleaning & Preprocessing
# ----------------------------------------------------------------------------
elif page == "3 · Cleaning & Preprocessing":
    st.title("🧼 Cleaning & Preprocessing Summary")

    st.markdown("### Step 1 — Removing duplicates")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='pastel-card' style='text-align:center'><h3>{n_before:,}</h3>Rows before</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='pastel-card' style='text-align:center'><h3>{n_before - n_after:,}</h3>Duplicates removed</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='pastel-card' style='text-align:center'><h3>{n_after:,}</h3>Rows after</div>", unsafe_allow_html=True)
    st.caption("Keeping duplicates would let the model see the same patient multiple times, leading to overfitting "
               "and inflated performance numbers.")

    st.markdown("### Step 2 — Missing values")
    missing_total = int(df.isnull().sum().sum())
    if missing_total == 0:
        st.markdown(
            "<span class='metric-pill pill-mint'>✓ No missing values found</span>",
            unsafe_allow_html=True,
        )
    st.caption("If there had been missing values: median imputation for Age, mode imputation for the binary columns.")

    st.markdown("### Step 3 — Encoding categorical features")
    st.markdown(
        "<div class='pastel-card'>Every column in this dataset is already numeric — symptoms and risk factors "
        "are 0/1 flags, and <code>Gender</code> is already 0 (Female) / 1 (Male). No one-hot or label encoding "
        "was needed.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Step 4 — Feature scaling (before / after)")
    st.caption("`StandardScaler` was fit on the training data only, then applied to both train and test, "
               "so every feature has mean ≈ 0 and standard deviation ≈ 1.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Before scaling — `Age`**")
        st.dataframe(pipe["X_train"]["Age"].describe()[["mean", "std", "min", "max"]].round(2).to_frame("value"))
    with col2:
        st.markdown("**After scaling — `Age`**")
        st.dataframe(pipe["X_train_scaled"]["Age"].describe()[["mean", "std", "min", "max"]].round(2).to_frame("value"))

    st.markdown(
        "<div class='pastel-card'>Scaling matters because <code>Age</code> (roughly 20–84) sits on a completely "
        "different scale than the binary 0/1 features. Without scaling, distance-based models (KNN, SVM) and "
        "gradient-based models (Logistic Regression) would let Age dominate just because of its larger numeric "
        "range — not because it's more predictive.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Step 5 — PCA check")
    st.markdown(
        "<div class='pastel-card'>PCA was tested as a dimensionality-reduction step, but it needed <b>16 of 18</b> "
        "components to reach 90% of the variance — almost no compression. The binary symptom/risk-factor columns "
        "are largely independent of each other, so PCA had little redundancy to squeeze out. "
        "<b>Conclusion: PCA was not used</b> — modeling was done on the original scaled features.</div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# 4. Model Results
# ----------------------------------------------------------------------------
elif page == "4 · Model Results":
    st.title("📊 Model Results")

    st.markdown(
        "<div class='pastel-card'>Two models suited to this binary classification task were trained and "
        "compared on the held-out test set: <b>Logistic Regression</b> (simple, fast, interpretable) and "
        "<b>Random Forest</b> (captures non-linear patterns).</div>",
        unsafe_allow_html=True,
    )

    results = pipe["results"]
    metric_rows = []
    for name, r in results.items():
        metric_rows.append({
            "Model": name,
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-score": r["f1"],
        })
    metrics_df = pd.DataFrame(metric_rows).set_index("Model").round(4)
    st.markdown("### Performance metrics")
    st.dataframe(metrics_df.style.format("{:.4f}").background_gradient(cmap="PuBu", axis=0), use_container_width=True)

    st.markdown("### Confusion matrices")
    col1, col2 = st.columns(2)
    cmap_choices = {"Logistic Regression": "PuBu", "Random Forest": "BuGn"}
    for col, (name, r) in zip([col1, col2], results.items()):
        with col:
            st.markdown(f"**{name}**")
            fig, ax = plt.subplots(figsize=(4.2, 4))
            disp = ConfusionMatrixDisplay(r["cm"], display_labels=["No Risk", "At Risk"])
            disp.plot(ax=ax, cmap=cmap_choices[name], colorbar=False)
            st.pyplot(fig)

    best_model = metrics_df["F1-score"].idxmax()
    st.markdown(
        f"<div class='pastel-card'><b>Best overall (by F1-score): {best_model}.</b> "
        "Both models score around 99% on every metric — driven by how strongly several symptoms and risk "
        "factors correlate with the target in this dataset. Random Forest reaches 100% accuracy on the "
        "training set, a mild sign of overfitting, though its test performance stays close to Logistic "
        "Regression's.</div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# 5. Interactive Prediction
# ----------------------------------------------------------------------------
elif page == "5 · Try a Live Prediction":
    st.title("🔮 Try a Live Prediction")

    st.markdown(
        "<div class='pastel-card'>Pick a model, enter a patient's symptoms and risk factors, "
        "and get a live heart-risk prediction.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Step 1 — Choose a model")
    model_choice = st.radio(
        "Model",
        ["🌳 Random Forest (Tree-based)", "📈 Logistic Regression"],
        horizontal=True,
        label_visibility="collapsed",
    )
    model_key = "Random Forest" if "Random Forest" in model_choice else "Logistic Regression"
    chosen_model = pipe["results"][model_key]["model"]

    st.markdown("---")
    st.markdown("### Step 2 — Enter patient information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", min_value=20, max_value=85, value=50)
        gender = st.selectbox("Gender", ["Female", "Male"])
        st.markdown("**Symptoms**")
        symptom_values = {}
        for c in SYMPTOM_COLS:
            symptom_values[c] = st.checkbox(FEATURE_INFO[c], key=c)

    with col2:
        st.markdown("**Risk factors**")
        risk_values = {}
        for c in RISK_FACTOR_COLS:
            risk_values[c] = st.checkbox(FEATURE_INFO[c], key=c)

    st.markdown("---")

    if st.button("✨ Predict heart risk"):
        input_dict = {c: int(v) for c, v in symptom_values.items()}
        input_dict.update({c: int(v) for c, v in risk_values.items()})
        input_dict["Gender"] = 1 if gender == "Male" else 0
        input_dict["Age"] = age

        input_df = pd.DataFrame([input_dict])[pipe["feature_cols"]]
        input_scaled = pd.DataFrame(
            pipe["scaler"].transform(input_df), columns=input_df.columns
        )

        pred = chosen_model.predict(input_scaled)[0]
        proba = chosen_model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.markdown(
                f"""
                <div class="pastel-card" style="background:#FFD9EC; text-align:center;">
                <h2>⚠️ At Risk</h2>
                <p style="font-size:1.1rem;">Estimated probability of heart risk: <b>{proba*100:.1f}%</b></p>
                <p style="color:#6E6680;">Model used: {model_key}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="pastel-card" style="background:#CFF3E1; text-align:center;">
                <h2>✅ No Risk</h2>
                <p style="font-size:1.1rem;">Estimated probability of heart risk: <b>{proba*100:.1f}%</b></p>
                <p style="color:#6E6680;">Model used: {model_key}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.caption("This is a demo built on a synthetic dataset — not a medical diagnosis tool.")
