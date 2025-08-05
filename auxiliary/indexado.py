import pathlib
import pandas as pd
import numpy as np
import faiss
from scipy import sparse
from sentence_transformers import SentenceTransformer, util
import os

def get_doc_top_tpcs(doc_distr, topn=10):
    sorted_tpc_indices = np.argsort(doc_distr)[::-1]
    top = sorted_tpc_indices[:topn].tolist()
    return [(k, doc_distr[k]) for k in top if doc_distr[k] > 0]

# Configuration
model_name = "quora-distilbert-multilingual"
embedding_size = 768  # Dimension of sentence embeddings
min_clusters = 8  # Minimum number of clusters for small topic sizes
top_k_hits = 10  # Number of nearest neighbors to retrieve
BATCH_SIZE = 32
THR_TOPIC_ASSIGNMENT = 0 #0.05
top_k = 10

# Load SentenceTransformer model
model = SentenceTransformer(model_name)

# Paths
PATH_SOURCE = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/26_jan/df_1.parquet")
PATH_MODEL = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/28_jan/poly_rosie_1_30")

for LANG in ["EN", "ES"]:
    FAISS_SAVE_DIR = pathlib.Path(f"INDICES/NEW_MODEL/faiss_indices_28_jan_15tpc_{LANG}")

    # Ensure save directory exists
    os.makedirs(FAISS_SAVE_DIR, exist_ok=True)

    # Load data
    print("-- Loading data...")
    raw = pd.read_parquet(PATH_SOURCE)
    thetas = sparse.load_npz(PATH_MODEL / "mallet_output" / f"thetas_{LANG}.npz").toarray()

    # Filter language documents
    topn = int(thetas.shape[1] / 3)
    raw = raw[raw.doc_id.str.contains(LANG)].copy()
    raw["thetas"] = list(thetas)
    raw["top_k"] = raw["thetas"].apply(lambda x: get_doc_top_tpcs(x, topn=topn))

    print("-- Checking existing indices...")
    topic_indices = {}
    all_indices_exist = True 

    for topic in range(thetas.shape[1]):
        index_path = FAISS_SAVE_DIR / f"faiss_index_topic_{topic}.index"
        doc_ids_path = FAISS_SAVE_DIR / f"doc_ids_topic_{topic}.npy"

        if index_path.exists() and doc_ids_path.exists():
            # Load the FAISS index and document IDs
            print(f"Loading indices for topic {topic}...")
            index = faiss.read_index(str(index_path))
            doc_ids = np.load(doc_ids_path, allow_pickle=True)
            topic_indices[topic] = {"index": index, "doc_ids": doc_ids}
        else:
            # If any index is missing, set flag to False
            print(f"Missing indices for topic {topic}")
            all_indices_exist = False
            break

    # We only generate embeddings and create new indices if any of the existing indices are missing
    if not all_indices_exist:
        print("-- Generating embeddings...")
        corpus_embeddings = model.encode(
            raw["text"].tolist(), show_progress_bar=True, convert_to_numpy=True, batch_size=BATCH_SIZE
        )
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

        # Create FAISS indices for each topic
        for topic in range(thetas.shape[1]):
            index_path = FAISS_SAVE_DIR / f"faiss_index_topic_{topic}.index"
            doc_ids_path = FAISS_SAVE_DIR / f"doc_ids_topic_{topic}.npy"

            if index_path.exists() and doc_ids_path.exists():
                continue

            print(f"-- Creating index for topic {topic}...")
            topic_embeddings = []
            doc_ids = []

            for i, top_k in enumerate(raw["top_k"]):
                for t, weight in top_k:
                    if t == topic and weight > THR_TOPIC_ASSIGNMENT:  # Relevance threshold for topic assignment
                        topic_embeddings.append(corpus_embeddings[i])
                        doc_ids.append(raw.iloc[i].doc_id)

            if topic_embeddings:
                topic_embeddings = np.array(topic_embeddings)
                N = len(topic_embeddings)
                n_clusters = max(int(4 * np.sqrt(N)), min_clusters)

                print(f"-- TOPIC {topic}: {N} documents, {n_clusters} clusters")

                # Train IVF index
                quantizer = faiss.IndexFlatIP(embedding_size)
                index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
                index.train(topic_embeddings)
                index.add(topic_embeddings)

                # Save the index and document IDs
                faiss.write_index(index, str(index_path))
                np.save(doc_ids_path, np.array(doc_ids))
                topic_indices[topic] = {"index": index, "doc_ids": doc_ids}
    else:
        print("-- All indices are loaded.")
        


# Query processing
def query_faiss(question, theta_query, top_k=10):
    """
    Perform a weighted topic search using FAISS indices.
    """
    question_embedding = model.encode([question], normalize_embeddings=True)[0]
    results = []

    for topic, weight in theta_query:
        index_path = FAISS_SAVE_DIR / f"faiss_index_topic_{topic}.index"
        doc_ids_path = FAISS_SAVE_DIR / f"doc_ids_topic_{topic}.npy"

        if index_path.exists() and doc_ids_path.exists():
            # Load the FAISS index and document IDs
            index = faiss.read_index(str(index_path))
            doc_ids = np.load(doc_ids_path, allow_pickle=True)

            # Perform the search
            distances, indices = index.search(np.expand_dims(question_embedding, axis=0), top_k)
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1:
                    results.append({"topic": topic, "doc_id": doc_ids[idx], "score": dist * weight}) #* weight

    # Sort results by score
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]