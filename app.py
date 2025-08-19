import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Text Clustering (Embeddings)", layout="wide")

st.title("🔎 Text Clustering with Hugging Face Embeddings")
st.markdown("Upload a CSV with a **text** column or use the sample data.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv("sample_data.csv")

assert 'text' in df.columns, "CSV must have a 'text' column"
texts = df['text'].astype(str).tolist()
texts = [t.strip().lower() for t in texts]

model_name = st.sidebar.selectbox(
    "Embedding model",
    [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "intfloat/multilingual-e5-base",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
)
k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=min(12, max(2, len(texts)//2)), value=6)
st.sidebar.caption("Tip: try different k, check silhouette score.")

with st.spinner("Computing embeddings..."):
    model = SentenceTransformer(model_name)
    X = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

with st.spinner("Clustering..."):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    labels = km.labels_
    sil = silhouette_score(X, labels)

st.success(f"Done. Silhouette score: **{sil:.3f}**")

df_out = df.copy(); df_out['cluster'] = labels
st.dataframe(df_out)

# Top terms per cluster
vect = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
Xtf = vect.fit_transform([t.lower() for t in df_out['text'].astype(str)])
terms = np.array(vect.get_feature_names_out())
st.subheader("Top terms per cluster")
for lab in sorted(df_out['cluster'].unique()):
    idx = np.where(labels==lab)[0]
    mean_tfidf = np.asarray(Xtf[idx].mean(axis=0)).ravel()
    top_idx = mean_tfidf.argsort()[::-1][:8]
    st.markdown(f"**Cluster {lab}:** " + ", ".join(terms[top_idx]))

# PCA plot
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X)
fig, ax = plt.subplots(figsize=(6,5))
scatter = ax.scatter(coords[:,0], coords[:,1], c=labels)
ax.set_title("PCA Projection")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
st.pyplot(fig)
