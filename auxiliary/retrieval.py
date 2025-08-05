import ast
import pathlib
import time
import pandas as pd
import numpy as np
import faiss
from scipy import sparse
from sentence_transformers import SentenceTransformer, util
import os
from tqdm import tqdm
from kneed import KneeLocator
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

def get_doc_top_tpcs(doc_distr, topn=10):
    sorted_tpc_indices = np.argsort(doc_distr)[::-1]
    top = sorted_tpc_indices[:topn].tolist()
    return [(k, doc_distr[k]) for k in top if doc_distr[k] > 0]

#TODO: dynamic thr here
def get_thresholds(mat_, poly_degree=3, smoothing_window=5):
    
    thrs = []
    for k in range(len(mat_.T)):
        allvalues = np.sort(mat_[:, k].flatten())
        step = int(np.round(len(allvalues) / 1000))
        x_values = allvalues[::step]
        y_values = (100 / len(allvalues)) * np.arange(0, len(allvalues))[::step]
        y_values_smooth = uniform_filter1d(y_values, size=smoothing_window)
        kneedle = KneeLocator(x_values, y_values_smooth, curve='concave', direction='increasing', interp_method='polynomial', polynomial_degree=poly_degree)
        thrs.append(kneedle.elbow)
    return thrs
            
# Configuration
model_name = "quora-distilbert-multilingual"
embedding_size = 768  # Dimension of sentence embeddings
min_clusters = 8  # Minimum number of clusters
top_k_hits = 10  # Number of nearest neighbors to retrieve
BATCH_SIZE = 32
THR_TOPIC_ASSIGNMENT = 0 #0.1
top_k = 10

# Load SentenceTransformer model
model = SentenceTransformer(model_name)

# Paths
NR_TPCS = 30
PATH_SOURCE = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/26_jan/df_1.parquet")
PATH_MODEL = pathlib.Path(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/26_jan_no_dup/poly_rosie_1_{NR_TPCS}")
LANG = "ES"
FAISS_SAVE_DIR = pathlib.Path(f"faiss_indices_28_jan_{NR_TPCS}tpc_{LANG}")
os.makedirs(FAISS_SAVE_DIR, exist_ok=True)

# Load data
print("-- Loading data...")
raw = pd.read_parquet(PATH_SOURCE)
thetas = sparse.load_npz(PATH_MODEL / "mallet_output" / f"thetas_{LANG}.npz").toarray()

raw_en = raw[raw.doc_id.str.contains("EN")].copy()
raw = raw[raw.doc_id.str.contains(LANG)].copy()
raw["thetas"] = list(thetas)
raw["top_k"] = raw["thetas"].apply(lambda x: get_doc_top_tpcs(x, topn=10))

thetas_en = sparse.load_npz(PATH_MODEL / "mallet_output" / f"thetas_EN.npz").toarray()
raw_en["thetas"] = list(thetas_en)
raw_en["top_k"] = raw_en["thetas"].apply(lambda x: get_doc_top_tpcs(x, topn=10))

# Generate embeddings if needed
print("-- Generating embeddings...")
CORPUS_EMBEDDINGS_PATH = FAISS_SAVE_DIR / "corpus_embeddings.npy"

if CORPUS_EMBEDDINGS_PATH.exists():
    print("-- Loading existing corpus embeddings...")
    corpus_embeddings = np.load(CORPUS_EMBEDDINGS_PATH)
else:
    print("-- Generating embeddings...")
    corpus_embeddings = model.encode(
        raw["text"].tolist(), show_progress_bar=True, convert_to_numpy=True, batch_size=BATCH_SIZE
    )
    corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    np.save(CORPUS_EMBEDDINGS_PATH, corpus_embeddings)  # Save embeddings

# ---- Create FAISS Index for Approximate Nearest Neighbors ----
print("-- Creating FAISS index for ANN...")
FAISS_INDEX_PATH = FAISS_SAVE_DIR / "faiss_index_IVF.index"

if FAISS_INDEX_PATH.exists():
    print("-- Loading existing FAISS index...")
    faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
else:
    print("-- Creating FAISS index for ANN...")
    n_clusters = 100  # Number of clusters
    quantizer = faiss.IndexFlatIP(embedding_size)
    faiss_index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
    
    faiss_index.train(corpus_embeddings)
    faiss_index.add(corpus_embeddings)
    faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))

# ---- Define Retrieval Methods ----

# 1. Exact Nearest Neighbors (Brute-force)
def exact_nearest_neighbors(query, corpus_embeddings, raw, top_k=10):
    time_start = time.time()
    query_embedding = model.encode([query], normalize_embeddings=True)
    cosine_similarities = np.dot(corpus_embeddings, query_embedding.T).squeeze()
    top_k_indices = np.argsort(-cosine_similarities)[:top_k]
    time_end = time.time()
    return [{"doc_id": raw.iloc[i].doc_id, "score": cosine_similarities[i]} for i in top_k_indices], time_end - time_start

# 2. Approximate Nearest Neighbors (FAISS)
def approximate_nearest_neighbors(query, faiss_index, doc_ids, top_k=10):
    time_start = time.time()
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    distances, indices = faiss_index.search(np.expand_dims(query_embedding, axis=0), top_k)
    time_end = time.time()
    return [{"doc_id": doc_ids[idx], "score": dist} for dist, idx in zip(distances[0], indices[0]) if idx != -1], time_end - time_start

# 3. Topic-based Exact Search
def topic_based_exact_search(query, theta_query, corpus_embeddings, raw, top_k=10, thrs=None, do_weighting=True):
    time_start = time.time()
    query_embedding = model.encode([query], normalize_embeddings=True)

    results = []
    for topic, weight in theta_query:
        thr = thrs[topic] if thrs is not None else THR_TOPIC_ASSIGNMENT
        if weight > thr:
            # Reset index so it matches corpus_embeddings indexing
            raw_reset_index = raw.reset_index(drop=True)
            topic_docs = raw_reset_index[raw_reset_index["top_k"].apply(lambda x: any(t == topic for t, _ in x))]
            
            # Now use `.iloc` to safely index into corpus_embeddings
            topic_embeddings = corpus_embeddings[topic_docs.index.to_numpy()]
            
            if len(topic_embeddings) == 0:
                continue

            # Compute cosine similarity
            cosine_similarities = np.dot(topic_embeddings, query_embedding.T).squeeze()
            top_k_indices = np.argsort(-cosine_similarities)[:top_k]

            for i in top_k_indices:
                score = cosine_similarities[i] * weight if do_weighting else cosine_similarities[i]
                results.append({"topic": topic, "doc_id": topic_docs.iloc[i].doc_id, "score": score})

    time_end = time.time()

    # Remove duplicates, keeping the highest score
    unique_results = {}
    for result in results:
        doc_id = result["doc_id"]
        if doc_id not in unique_results or result["score"] > unique_results[doc_id]["score"]:
            unique_results[doc_id] = result

    return sorted(unique_results.values(), key=lambda x: x["score"], reverse=True)[:top_k], time_end - time_start

# 4. Topic-based Approximate Search (FAISS per topic)
def topic_based_approximate_search(query, theta_query, top_k=10, thr=None, do_weighting=True):
    time_start = time.time()
    query_embedding = model.encode([query], normalize_embeddings=True)[0]
    results = []
    for topic, weight in theta_query:
        thr = thrs[topic] if thrs is not None else THR_TOPIC_ASSIGNMENT
        if weight > thr:
            index_path = FAISS_SAVE_DIR / f"faiss_index_topic_{topic}.index"
            doc_ids_path = FAISS_SAVE_DIR / f"doc_ids_topic_{topic}.npy"
            if index_path.exists() and doc_ids_path.exists():
                index = faiss.read_index(str(index_path))
                doc_ids = np.load(doc_ids_path, allow_pickle=True)
                distances, indices = index.search(np.expand_dims(query_embedding, axis=0), top_k)
                for dist, idx in zip(distances[0], indices[0]):
                    if idx != -1:
                        score = dist * weight if do_weighting else dist
                        results.append({"topic": topic, "doc_id": doc_ids[idx], "score": score})
                    
    # Remove duplicates, keeping the highest score
    unique_results = {}
    for result in results:
        doc_id = result["doc_id"]
        if doc_id not in unique_results or result["score"] > unique_results[doc_id]["score"]:
            unique_results[doc_id] = result
    time_end = time.time()
    return sorted(unique_results.values(), key=lambda x: x["score"], reverse=True)[:top_k], time_end - time_start

# ---- Testing the Four Methods ----
paths_ = os.listdir("GENERATIONS/OLD_MODEL/questions_queries")
for path_queries in paths_:

    df_q = pd.read_excel("GENERATIONS/OLD_MODEL/questions_queries/" + path_queries)

    # Calculate threshold dynamically
    thetas_es = sparse.load_npz(PATH_MODEL / "mallet_output" / "thetas_ES.npz").toarray()
    thrs_ = get_thresholds(thetas_es, poly_degree=3, smoothing_window=5)
    if "llama" in path_queries:
        thrs_keep = [thrs_]
    else:
        thrs_keep = [None, thrs_]
    for thrs in thrs_keep:
        
        print(f"Calculating results with thresholds: {thrs}")
        save_thr = "_dynamic" if thrs is not None else ""

        # initialize columns to store results
        for key_results in ["results_1", "results_2", "results_3_weighted", "results_3_unweighted", "results_4_weighted", "results_4_unweighted", "time_1", "time_2", "time_3", "time_4", "theta_10"]:
            df_q[key_results] = None
        for id_row, row in tqdm(df_q.iterrows(), total=df_q.shape[0]):
            if NR_TPCS != 30:
                row[f"theta_{NR_TPCS}"] = raw_en[raw_en.doc_id == row.doc_id].thetas.values[0]
                row[f"top_k_{NR_TPCS}"] = raw_en[raw_en.doc_id == row.doc_id].top_k.values[0]
            queries = ast.literal_eval(row.queries)
            
            if NR_TPCS == 30:
                theta_query = ast.literal_eval(row.top_k)
            else:
                theta_query = row[f"top_k_{NR_TPCS}"]
            results_1 = []
            results_2 = []
            results_3_weighted = []
            results_3_unweighted = []
            results_4_weighted = []
            results_4_unweighted = []
            
            time_1 = []
            time_2 = []
            time_3 = []
            time_4 = []
            for query in queries:
                r1, t1 = exact_nearest_neighbors(query, corpus_embeddings, raw, top_k)
                r2, t2 = approximate_nearest_neighbors(query, faiss_index, raw["doc_id"].tolist(), top_k)
                
                r3_w, t3 = topic_based_exact_search(query, theta_query, corpus_embeddings, raw, top_k, thrs, do_weighting=True)
                r3_unw, _ = topic_based_exact_search(query, theta_query, corpus_embeddings, raw, top_k,thrs,  do_weighting=False)
                r4_w, t4 = topic_based_approximate_search(query, theta_query, top_k, thrs, do_weighting=True)
                r4_unw, _ = topic_based_approximate_search(query, theta_query, top_k, thrs, do_weighting=False)
                
                results_1.append(r1)
                results_2.append(r2)
                results_3_weighted.append(r3_w)
                results_3_unweighted.append(r3_unw)
                results_4_weighted.append(r4_w)
                results_4_unweighted.append(r4_unw)
                
                time_1.append(t1)
                time_2.append(t2)
                time_3.append(t3)
                time_4.append(t4)
                
                # print comparison of times
                print(f"Exact NN: {t1:.2f}s, Approx NN: {t2:.2f}s, Topic-based Exact: {t3:.2f}s, Topic-based Approx: {t4:.2f}s")
                
            df_q.at[id_row, "results_1"] = results_1
            df_q.at[id_row, "results_2"] = results_2
            df_q.at[id_row, "results_3_weighted"] = results_3_weighted
            df_q.at[id_row, "results_3_unweighted"] = results_3_unweighted
            df_q.at[id_row, "results_4_weighted"] = results_4_weighted
            df_q.at[id_row, "results_4_unweighted"] = results_4_unweighted
            #import pdb; pdb.set_trace()

            # Store time as a single value
            df_q.at[id_row, "time_1"] = np.mean(time_1)
            df_q.at[id_row, "time_2"] = np.mean(time_2)
            df_q.at[id_row, "time_3"] = np.mean(time_3)
            df_q.at[id_row, "time_4"] = np.mean(time_4)
            
        # all (results_1, results_2, results_3_weighted, results_3_unweighted, results_4_weighted, results_4_unweighted) are of the same length. Explode the dataframe so each row contains a single query and its results

        # Save the dataframe
        path_save = "GENERATIONS/OLD_MODEL/relevant/" + path_queries.replace(".xlsx", f"_results_model{NR_TPCS}tpc_thr_{save_thr}.parquet")
        df_q.to_parquet(path_save)
                
        
        # Convert all result lists to individual rows in one step
        columns_to_explode = ["results_1", "results_2", "results_3_weighted", "results_3_unweighted", "results_4_weighted", "results_4_unweighted"]
        df_q = df_q.explode(columns_to_explode, ignore_index=True)

        # Efficiently combine results without repeated parsing
        def combine_results(row):
            doc_ids = set()
            for col in columns_to_explode:
                try:
                    content = ast.literal_eval(row[col]) if isinstance(row[col], str) else row[col]
                except:
                    content = row[col]
                if isinstance(content, list):
                    doc_ids.update(doc["doc_id"] for doc in content)
            return list(doc_ids)

        df_q["all_results"] = df_q[columns_to_explode].apply(combine_results, axis=1)

        # Select only necessary columns
        df_q_eval = df_q[['pass_id', 'doc_id', 'passage', 'top_k', 'question', 'queries', 'all_results']].copy()

        # Explode only once
        df_q_eval = df_q_eval.explode("all_results", ignore_index=True)

        # Use `map` instead of `apply` for better performance
        doc_map = raw.set_index("doc_id")["text"].to_dict()
        df_q_eval["all_results_content"] = df_q_eval["all_results"].map(doc_map)

        # Save the processed dataframe
        path_save = "GENERATIONS/OLD_MODEL/relevant/" + path_queries.replace(".xlsx", f"_results_model{NR_TPCS}tpc_thr_{save_thr}_combined_to_retrieve_relevant.parquet")
        df_q_eval.to_parquet(path_save)
