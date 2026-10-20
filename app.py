from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")

HERE = Path(__file__).parent
COLS = ["Annual Income (k$)", "Spending Score (1-100)"]

ACTIONS = {
    "Standard customers": "Regular promotions to keep them engaged.",
    "High income, high spending": "Premium offers, loyalty programme, early access.",
    "Young, low income, high spending": "Affordable trendy products, bundles and discounts.",
    "High income, low spending": "Personalised premium offers to raise their spending.",
    "Low income, low spending": "Low-cost value deals; keep marketing spend small.",
}


@st.cache_resource
def load_model():
    scaler = joblib.load(HERE / "scaler.joblib")
    model = joblib.load(HERE / "kmeans_model.joblib")
    names = joblib.load(HERE / "segment_names.joblib")
    return scaler, model, names


@st.cache_data
def load_data():
    path = HERE / "customer_segmentation data.csv"
    return pd.read_csv(path) if path.exists() else None


try:
    scaler, model, names = load_model()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Put scaler.joblib, kmeans_model.joblib "
             "and segment_names.joblib in the same folder as app.py.")
    st.stop()

st.title("🛍️ Customer Segmentation")
st.caption("KMeans (5 clusters) on annual income and spending score. "
           "Final model silhouette score: 55.47.")

tab1, tab2, tab3 = st.tabs(["Predict one customer", "Segment a CSV", "About the segments"])

# ---------- Tab 1: single prediction ----------
with tab1:
    left, right = st.columns([1, 2])
    with left:
        with st.form("predict_form"):
            income = st.number_input("Annual income (k$)", 0, 200, 60)
            spend = st.slider("Spending score (1-100)", 1, 100, 50)
            submitted = st.form_submit_button("Submit")

        if submitted:
            new = pd.DataFrame([[income, spend]], columns=COLS)
            cluster = int(model.predict(scaler.transform(new))[0])
            segment = names[cluster]
            st.success(f"**Cluster {cluster}: {segment}**")
            st.write(ACTIONS.get(segment, ""))

    with right:
        data = load_data()
        if data is not None:
            data = data.copy()
            data["Cluster"] = model.predict(scaler.transform(data[COLS]))
            data["Segment"] = data["Cluster"].map(names)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=data, x=COLS[0], y=COLS[1], hue="Segment",
                            palette="tab10", ax=ax, alpha=0.8)
            if submitted:
                ax.scatter(income, spend, marker="*", s=400, c="black", label="This customer")
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            st.pyplot(fig)
        else:
            st.info("Add 'customer_segmentation data.csv' next to app.py to see the chart.")

# ---------- Tab 2: batch ----------
with tab2:
    st.write("Upload a CSV with the columns **Annual Income (k$)** and "
             "**Spending Score (1-100)**.")
    file = st.file_uploader("CSV file", type="csv")
    if file:
        batch = pd.read_csv(file)
        missing = [c for c in COLS if c not in batch.columns]
        if missing:
            st.error(f"Missing column(s): {', '.join(missing)}")
        else:
            batch["Cluster"] = model.predict(scaler.transform(batch[COLS]))
            batch["Segment"] = batch["Cluster"].map(names)
            st.dataframe(batch)
            st.bar_chart(batch["Segment"].value_counts())
            st.download_button("Download results", batch.to_csv(index=False).encode("utf-8"),
                               "customers_with_segments.csv", "text/csv")

# ---------- Tab 3: segments ----------
with tab3:
    rows = []
    for cid in sorted(names):
        rows.append({"Cluster": cid, "Segment": names[cid],
                     "Suggested action": ACTIONS.get(names[cid], "")})
    st.table(pd.DataFrame(rows).set_index("Cluster"))
    st.write("Age and gender were not used to build the clusters; they are only "
             "used to describe the segments. With all four features the clusters were "
             "mostly split by gender and the silhouette score was only 27.19.")
