import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

try:
    import umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

from sentence_transformers import SentenceTransformer

def load_texts(path):
    df = pd.read_csv(path)
    assert 'text' in df.columns, "CSV must have a 'text' column"
    return df['text'].astype(str).tolist(), df

def preprocess(texts):
    return [t.strip().lower() for t in texts]

def embed(texts, model_name):
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return np.array(emb)

def kmeans_cluster(X, k, seed=42):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = kmeans.fit_predict(X)
    return kmeans, labels

def best_k_by_silhouette(X, kmin=2, kmax=10, seed=42):
    best_k, best_score = None, -1.0
    for k in range(kmin, kmax + 1):
        km, labels = kmeans_cluster(X, k, seed=seed)
        score = silhouette_score(X, labels)
        print(f"k={k}, silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score

def top_terms_per_cluster(texts, labels, topn=8):
    vect = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vect.fit_transform(texts)
    terms = np.array(vect.get_feature_names_out())
    tops = {}
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) == 0: 
            tops[lab] = []
            continue
        mean_tfidf = np.asarray(X[idx].mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[::-1][:topn]
        tops[lab] = terms[top_idx].tolist()
    return tops

def visualize_pca(X, labels, out_path="plot_pca.png"):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    plt.figure(figsize=(7,6))
    scatter = plt.scatter(coords[:,0], coords[:,1], c=labels)
    plt.title("PCA Projection of Embeddings")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path

def visualize_umap(X, labels, out_path="plot_umap.png"):
    if not HAVE_UMAP:
        return None
    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(X)
    plt.figure(figsize=(7,6))
    plt.scatter(coords[:,0], coords[:,1], c=labels)
    plt.title("UMAP Projection of Embeddings")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return out_path

def save_clusters(df, labels, out_csv="clusters.csv", samples_per_cluster=3):
    df = df.copy()
    df['cluster'] = labels
    df.to_csv(out_csv, index=False)
    # Print sample texts per cluster
    for lab in sorted(df['cluster'].unique()):
        subset = df[df['cluster'] == lab]['text'].head(samples_per_cluster).tolist()
        print(f"\n=== Cluster {lab} ===")
        for s in subset:
            print(f"- {s[:140]}{'...' if len(s)>140 else ''}")
    return out_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='sample_data.csv', help='CSV with a column named text')
    ap.add_argument('--k', type=int, default=0, help='Number of clusters. 0 = auto by silhouette (2..10)')
    ap.add_argument('--model', type=str, default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    ap.add_argument('--outdir', type=str, default='.', help='Output directory')
    args = ap.parse_args()

    texts, df = load_texts(args.data)
    texts_prep = preprocess(texts)
    X = embed(texts_prep, args.model)

    k = args.k
    if k <= 1:
        k, score = best_k_by_silhouette(X, 2, min(10, max(2, len(texts)//3)))
        print(f"[Auto] best k by silhouette = {k} (score={score:.4f})")

    km, labels = kmeans_cluster(X, k)
    tops = top_terms_per_cluster(texts_prep, labels, topn=8)
    print("\nTop terms per cluster:")
    for lab in sorted(tops.keys()):
        print(f"Cluster {lab}: {', '.join(tops[lab])}")

    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, 'clusters.csv')
    save_clusters(df, labels, out_csv=out_csv, samples_per_cluster=4)

    pca_path = os.path.join(args.outdir, 'plot_pca.png')
    visualize_pca(X, labels, out_path=pca_path)
    umap_path = os.path.join(args.outdir, 'plot_umap.png')
    if HAVE_UMAP:
        visualize_umap(X, labels, out_path=umap_path)

    # Optional: FAISS nearest neighbor demo
    try:
        import faiss
        index = faiss.IndexFlatIP(X.shape[1])  # cosine if normalized
        index.add(X.astype('float32'))
        D, I = index.search(X[:5].astype('float32'), 6)
        print("\nFAISS nearest neighbors (first 5 texts):")
        for i, (drow, irow) in enumerate(zip(D, I)):
            print(f"Query {i}:")
            for rank, (dist, idx) in enumerate(zip(drow, irow)):
                print(f"  {rank:>2}. idx={idx}, sim={dist:.3f}, cluster={labels[idx]}")
    except Exception as e:
        print("[FAISS demo skipped]:", e)

    print(f"\nArtifacts saved to: {os.path.abspath(args.outdir)}")
    print(f"- Clusters CSV: {out_csv}")
    print(f"- PCA plot: {pca_path}")
    if HAVE_UMAP:
        print(f"- UMAP plot: {umap_path}")

if __name__ == '__main__':
    main()
