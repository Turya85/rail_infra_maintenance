import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Railway Failure Dashboard", page_icon="🚦", layout="wide")
st.title("Railway Infrastructure – Failure‑Risk Dashboard")

# -------------------------------------------------
# 1️⃣  FILE UPLOAD (MANDATORY)
# -------------------------------------------------
st.sidebar.header("Upload CSV first")
file = st.sidebar.file_uploader("Select your railway_infra_component_data.csv", type=["csv"])

if file is None:
    st.info("⬆️  Please upload a CSV to begin the analysis. The dashboard will refresh automatically once a file is selected.")
    st.stop()

# The moment a file is provided, proceed

df_raw = pd.read_csv(file)

# -------------------------------------------------
# 2️⃣  DATA PREP – calculate Risk_Score & Failed flag
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def prep(df):
    rnd = np.random.normal(0, 0.1, size=len(df))
    risk = (df["Age_Years"] / 15) + df["Maintenance_Count"] * 0.1 + rnd
    df = df.copy()
    df["Risk_Score"] = np.clip(risk.round(2), 0, 1)
    df["Failed"] = (df["Risk_Score"] > 0.70).astype(int)
    return df

df = prep(df_raw)

# -------------------------------------------------
# 3️⃣  KPI CARDS
# -------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Components", f"{len(df):,}")
col2.metric("Failure Rate", f"{df['Failed'].mean()*100:.1f}%")
col3.metric("Avg Risk Score", f"{df['Risk_Score'].mean():.2f}")

st.markdown("---")

# Helper to display two figures side‑by‑side

def two_cols(fig_a, caption_a, fig_b, caption_b):
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(fig_a); st.caption(caption_a)
    with c2:
        st.pyplot(fig_b); st.caption(caption_b)

# -------------------------------------------------
# 4️⃣  VISUALS
# -------------------------------------------------
# A. Failure rate by component & zone
rate_comp = df.groupby("Component")["Failed"].mean().sort_values()
fig1, ax1 = plt.subplots(figsize=(6,4)); rate_comp.plot.barh(ax=ax1, color="steelblue"); ax1.set_xlabel("Failure rate"); ax1.set_title("By component")
rate_zone = df.groupby("Zone")["Failed"].mean().sort_values()
fig2, ax2 = plt.subplots(figsize=(6,4)); rate_zone.plot.bar(ax=ax2, color="indianred"); ax2.set_ylabel("Failure rate"); ax2.set_title("By zone"); plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
two_cols(fig1, "Component‑wise", fig2, "Zone‑wise")

# B. Risk score hist + scatter Age‑Risk
fig3, ax3 = plt.subplots(figsize=(6,4)); sns.histplot(df, x="Risk_Score", hue="Failed", bins=20, kde=True, ax=ax3); ax3.set_title("Risk score distribution")
fig4, ax4 = plt.subplots(figsize=(6,4)); sns.scatterplot(data=df, x="Age_Years", y="Risk_Score", hue="Failed", alpha=0.6, ax=ax4); ax4.set_title("Age vs Risk score")
two_cols(fig3, "Histogram", fig4, "Scatter plot")

# C. Failure rate by Category + Boxplot risk by Zone
rate_cat = df.groupby("Category")["Failed"].mean().sort_values()
fig5, ax5 = plt.subplots(figsize=(6,4)); rate_cat.plot.bar(ax=ax5, color="slateblue"); ax5.set_ylabel("Failure rate"); ax5.set_title("By category"); plt.setp(ax5.get_xticklabels(), rotation=45, ha="right")
fig6, ax6 = plt.subplots(figsize=(6,4)); sns.boxplot(data=df, x="Zone", y="Risk_Score", palette="Pastel1", ax=ax6); ax6.set_title("Risk by zone"); plt.setp(ax6.get_xticklabels(), rotation=45, ha="right")
two_cols(fig5, "Category", fig6, "Risk spread")

# D. Correlation heatmap
fig7, ax7 = plt.subplots(figsize=(10,6)); sns.heatmap(df.select_dtypes("number").corr(), annot=True, cmap="coolwarm", ax=ax7); ax7.set_title("Correlation heatmap"); st.pyplot(fig7)

st.markdown("---")

# -------------------------------------------------
# 5️⃣  FEATURE IMPORTANCE (Random Forest) – optional context
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def feature_importance(df):
    cat_cols = ["Component", "Zone", "Category", "Failure_Mode"]
    enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    X = enc.drop(["Risk_Score", "Failed"], axis=1)
    y = enc["Failed"]
    trX, teX, trY, teY = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(trX, trY)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]

imp = feature_importance(df)
fig8, ax8 = plt.subplots(figsize=(6,4)); imp[::-1].plot.barh(ax=ax8, color="seagreen"); ax8.set_xlabel("Importance"); ax8.set_title("Top 10 feature importances"); st.pyplot(fig8)

# -------------------------------------------------
# 6️⃣  SUMMARY TABLE & DOWNLOAD
# -------------------------------------------------
summary = (df.pivot_table(index="Zone", columns="Component", values="Failed", aggfunc="mean")*100).round(1)
st.subheader("Failure rate (%) by zone & component")
st.dataframe(summary)

@st.cache_data(show_spinner=False)
def to_excel_bytes(data):
    with io.BytesIO() as buf:
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            data.to_excel(xw, index=False, sheet_name="Analysis")
            summary.to_excel(xw, sheet_name="Zone_Component")
        return buf.getvalue()

st.download_button("📊 Download Excel report", data=to_excel_bytes(df), file_name="railway_failure_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# FOOTER
st.caption("Upload‑first workflow: dashboard appears only after you provide data. Numerical diagnostics omitted for presentation clarity.")
