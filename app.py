import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.express as px

# -----------------------------------
# 🎯 Page Configuration
# -----------------------------------
st.set_page_config(page_title="Clustering Playground", layout="wide")
st.title("🔍 Interactive Clustering Visualization App")

st.markdown("""
This app lets you explore **clustering algorithms** on synthetic datasets.
Experiment with different **scalers**, **algorithms**, and **parameters** to see how clusters form and change.
""")

# -----------------------------------
# 🧠 Step 1: Dataset Selection
# -----------------------------------
dataset_choice = st.sidebar.selectbox(
    "Choose Synthetic Dataset",
    ["make_blobs", "make_moons"]
)

if dataset_choice == "make_blobs":
    n_clusters = st.sidebar.slider("Number of Centers", 2, 8, 4)
    cluster_std = st.sidebar.slider("Cluster Standard Deviation", 0.5, 3.0, 1.0)
    X, y_true = make_blobs(n_samples=500, centers=n_clusters, cluster_std=cluster_std, random_state=42)
else:
    noise = st.sidebar.slider("Noise Level", 0.0, 0.2, 0.08)
    X, y_true = make_moons(n_samples=500, noise=noise, random_state=42)

df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])

# -----------------------------------
# ⚙️ Step 2: Feature Scaling
# -----------------------------------
st.sidebar.subheader("Feature Scaling")
scaler_choice = st.sidebar.selectbox("Select Scaler", ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"])

if scaler_choice == "StandardScaler":
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
elif scaler_choice == "MinMaxScaler":
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
elif scaler_choice == "RobustScaler":
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X

# -----------------------------------
# 🤖 Step 3: Algorithm Selection
# -----------------------------------
st.sidebar.subheader("Clustering Algorithm")
algorithm = st.sidebar.selectbox(
    "Choose Algorithm",
    ["K-Means", "DBSCAN", "Agglomerative"]
)

# Dynamic Hyperparameter Inputs
if algorithm == "K-Means":
    k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 4)
    model = KMeans(n_clusters=k, random_state=42)
elif algorithm == "DBSCAN":
    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5)
    min_samples = st.sidebar.slider("Min Samples", 3, 15, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)
elif algorithm == "Agglomerative":
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)
    linkage = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

# -----------------------------------
# 🚀 Step 4: Fit Model
# -----------------------------------
labels = model.fit_predict(X_scaled)
df["Cluster"] = labels

# -----------------------------------
# 📊 Step 5: Evaluation Metrics
# -----------------------------------
if len(set(labels)) > 1 and -1 not in labels:
    silhouette = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)

    metrics_df = pd.DataFrame({
        'Metric': ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Score'],
        'Value': [silhouette, dbi, ch]
    })
    st.subheader("📈 Clustering Performance Metrics")
    st.dataframe(metrics_df, use_container_width=True)

    # Dynamic Feedback
    if silhouette > 0.5:
        st.success("✅ Clusters are well-separated and meaningful.")
    elif silhouette > 0.3:
        st.warning("⚠️ Clusters are moderately distinct.")
    else:
        st.error("❌ Clusters overlap or are not meaningful.")
else:
    st.warning("Silhouette and other scores not applicable (only one cluster or noise detected).")

# -----------------------------------
# 🎨 Step 6: Visualization
# -----------------------------------
fig = px.scatter(
    df, x="Feature 1", y="Feature 2",
    color=df["Cluster"].astype(str),
    title=f"{algorithm} Clustering Results on {dataset_choice}",
    color_discrete_sequence=px.colors.qualitative.Bold
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# 🧾 Step 7: Cluster Summary
# -----------------------------------
if algorithm == "K-Means" and len(set(labels)) > 1:
    cluster_summary = pd.DataFrame(df.groupby("Cluster")[["Feature 1", "Feature 2"]].mean())
    st.subheader("📘 Cluster Centroids (Mean of Each Feature)")
    st.dataframe(cluster_summary)
