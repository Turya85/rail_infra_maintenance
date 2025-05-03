# streamlit_app.py
import streamlit as st
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Upload section
st.sidebar.header("Upload CSV")
user_file = st.sidebar.file_uploader("railway_infra_component_data.csv", type=["csv"])

if not user_file:
    st.info("Please upload a CSV to start.")
    st.stop()

df = pd.read_csv(user_file)      # <-- replaces hard‑coded CSV_PATH

# 2. Risk score + target
def compute_risk_score(r):
    return np.clip(round(r["Age_Years"]/15 + r["Maintenance_Count"]*0.1 + np.random.normal(0,0.1), 2), 0, 1)

df["Risk_Score"] = df.apply(compute_risk_score, axis=1)
df["Failed"] = (df["Risk_Score"] > 0.70).astype(int)

# 3. EDA
st.subheader("Basic info")
st.dataframe(df.head())
st.text(df.describe().to_string())

failure_by_comp = df.groupby("Component")["Failed"].mean().sort_values(ascending=False)
st.subheader("Failure rate by component")
st.bar_chart(failure_by_comp)

fig, ax = plt.subplots(figsize=(8,5))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# 4–5. Pre‑processing & model (identical to your script)
cat_cols = ["Component", "Zone", "Category", "Failure_Mode"]
df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
X = df_enc.drop(["Risk_Score", "Failed"], axis=1)
y = df_enc["Failed"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X_tr, y_tr)

# 6. Evaluation
y_pred = clf.predict(X_te)
st.subheader("Confusion matrix")
st.dataframe(pd.DataFrame(confusion_matrix(y_te, y_pred),
                          index=["Actual 0","Actual 1"],
                          columns=["Pred 0","Pred 1"]))
st.subheader("Classification report")
st.text(classification_report(y_te, y_pred, digits=3))
