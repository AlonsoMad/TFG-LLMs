import ast
import pathlib
import time
import pandas as pd
import numpy as np
from scipy import sparse
from sentence_transformers import SentenceTransformer, util
import os
from tqdm import tqdm
from kneed import KneeLocator
from scipy.ndimage import uniform_filter1d
import argparse
from prompter import Prompter
import re
import pathlib
import re
import ast

# Configuration
model_name = "quora-distilbert-multilingual"
embedding_size = 768  # Dimension of sentence embeddings
min_clusters = 8  # Minimum number of clusters
top_k_hits = 10  # Number of nearest neighbors to retrieve
BATCH_SIZE = 32
THR_TOPIC_ASSIGNMENT = 0
top_k = 10
######################
# PATHS TO TEMPLATES #
######################
# 3. ANSWER GENERATION
_3_INSTRUCTIONS_PATH = "templates/question_answering.txt"
_4_INSTRUCTIONS_PATH = "templates/discrepancy_detection.txt"
RELEVANCE_PROMPT = "templates/test_relevance.txt"

# Load SentenceTransformer model
model = SentenceTransformer(model_name)

# Paths
NR_TPCS = 30
PATH_SOURCE = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/26_jan/df_1.parquet")
PATH_MODEL = pathlib.Path(f"/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/26_jan_no_dup/poly_rosie_1_{NR_TPCS}")
LANG = "ES"
FAISS_SAVE_DIR = pathlib.Path(f"faiss_indices_28_jan_{NR_TPCS}tpc_{LANG}")
os.makedirs(FAISS_SAVE_DIR, exist_ok=True)

def get_doc_top_tpcs(doc_distr, topn=10):
    sorted_tpc_indices = np.argsort(doc_distr)[::-1]
    top = sorted_tpc_indices[:topn].tolist()
    return [(k, doc_distr[k]) for k in top if doc_distr[k] > 0]

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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_queries", type=str, required=True, help="Path to the file with questions and queries.")
    parser.add_argument("--path_relevant", type=str, required=True, help="Path to relevant document parquet files.")
    parser.add_argument("--path_save", type=str, required=True, help="Path to save results.")
    parser.add_argument("--method_eval", type=str, required=True, help="Method evaluation column name.")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K documents to consider.")
    parser.add_argument("--sample_eval", type=int, default=1, help="Number of samples to evaluate.")
    return parser.parse_args()

def extend_to_full_sentence(text: str, num_words: int) -> str:
    text_in_words = text.split()
    truncated_text = " ".join(text_in_words[:num_words])
    remaining_text = " ".join(text_in_words[num_words:])
    period_index = remaining_text.find(".")
    if period_index != -1:
        extended_text = f"{truncated_text} {remaining_text[:period_index + 1]}"
    else:
        extended_text = truncated_text
    return re.sub(r'\s([?.!,"])', r'\1', extended_text)

def main():
    args = parse_arguments()
    os.makedirs(args.path_save, exist_ok=True)
    
    ############################################################
    # Retrieve relevant documents
    ############################################################
    print("Retrieving relevant documents...")
    # Load all queries
    df_q_all = pd.read_excel(args.path_queries)
    
    # Load previous results if they exist
    path_prev = args.path_relevant + "/" + pathlib.Path(args.path_queries).name.replace(".xlsx", f"_results_model{NR_TPCS}tpc_thr__dynamic.parquet")
    if os.path.exists(path_prev):
        print("Loading previous results...")
        df_q_prev = pd.read_parquet(path_prev)
    
    # Remove already processed queries
    if "df_q_prev" in locals():
        df_q_all = df_q_all[~df_q_all["question_id"].isin(df_q_prev["question_id"])]
        
        print(f"Already processed {df_q_prev.shape[0]} queries. Remaining: {df_q_all.shape[0]}")
    
    # Sample queries
    df_q = df_q_all.sample(args.sample_eval, random_state=42)
    
    print("Keeping a sample of ", args.sample_eval, " queries.")
    
    print("-- Generating embeddings...")
    CORPUS_EMBEDDINGS_PATH = FAISS_SAVE_DIR / "corpus_embeddings.npy"

    if CORPUS_EMBEDDINGS_PATH.exists():
        print("-- Loading existing corpus embeddings...")
        corpus_embeddings = np.load(CORPUS_EMBEDDINGS_PATH)
    
    # Calculate threshold dynamically
    print("-- Calculating thresholds...")
    thetas_es = sparse.load_npz(PATH_MODEL / "mallet_output" / "thetas_ES.npz").toarray()
    thrs = get_thresholds(thetas_es, poly_degree=3, smoothing_window=5)
    
    print("-- Loading raw data...")
    raw = pd.read_parquet(PATH_SOURCE)
    thetas = sparse.load_npz(PATH_MODEL / "mallet_output" / f"thetas_{LANG}.npz").toarray()
    raw_en = raw[raw.doc_id.str.contains("EN")].copy()
    raw = raw[raw.doc_id.str.contains(LANG)].copy()
    raw["thetas"] = list(thetas)
    raw["top_k"] = raw["thetas"].apply(lambda x: get_doc_top_tpcs(x, topn=10))
    thetas_en = sparse.load_npz(PATH_MODEL / "mallet_output" / f"thetas_EN.npz").toarray()
    raw_en["thetas"] = list(thetas_en)
    raw_en["top_k"] = raw_en["thetas"].apply(lambda x: get_doc_top_tpcs(x, topn=10))
    
    print(f"Calculating results with thresholds: {thrs}")
    
    for key_results in [args.method_eval]:
        
        df_q[key_results] = None
        for id_row, row in tqdm(df_q.iterrows(), total=df_q.shape[0]):
            
            try:
                queries = ast.literal_eval(row.queries)
                
                theta_query = ast.literal_eval(row.top_k)
                
                results_3_weighted, time_3 = [], []

                for query in queries:
                    
                    r3_w, t3 = topic_based_exact_search(query, theta_query, corpus_embeddings, raw, top_k, thrs, do_weighting=True)
                    
                    results_3_weighted.append(r3_w)
                    time_3.append(t3)
                    
                    # print comparison of times
                    print(f"Topic-based Exact: {t3:.2f}s")

                df_q.at[id_row, "results_3_weighted"] = results_3_weighted
                df_q.at[id_row, "time_3"] = np.mean(time_3)
            except Exception as e:
                print(f"Error with question {row.question_id}: {e}")
                # remove row from dataframe
                df_q = df_q[df_q.question_id != row.question_id]

        # concat with previous results
        if "df_q_prev" in locals():
            df_q = pd.concat([df_q, df_q_prev], ignore_index=True)
        
        # Save the dataframe
        path_save = args.path_relevant + "/" + pathlib.Path(args.path_queries).name.replace(".xlsx", f"_results_model{NR_TPCS}tpc_thr__dynamic.parquet")
        df_q.to_parquet(path_save)
                
        
        # Convert all result lists to individual rows in one step
        columns_to_explode = ["results_3_weighted"]
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
        path_save = args.path_save + "/" + pathlib.Path(args.path_queries).name.replace(".xlsx", f"_results_model{NR_TPCS}tpc_thr_dynamic_combined_to_retrieve_relevant.parquet")
        df_q_eval.to_parquet(path_save)
    
    print("Retrieving completed!")
    #import pdb; pdb.set_trace()

    ############################################################
    # Retrieve answers
    ############################################################
    
    print("Answers will be saved in the following path: ", args.path_save)
    checkpoint_file = f"{args.path_save}/checkpoint.parquet"
    processed_questions = set()
    
    if os.path.exists(checkpoint_file):
        checkpoint_df = pd.read_parquet(checkpoint_file)
        processed_questions = set(checkpoint_df["question_id"].unique())
        print(f"Already processed {len(processed_questions)} processed_questions")
    paths_ = [p for p in os.listdir(args.path_relevant) if p.endswith("thr__dynamic.parquet")]
    print(f"Processing {len(paths_)} files...")
    
    path_queries = paths_[0]
    df = pd.read_parquet(os.path.join(args.path_relevant, path_queries))
    df = df.drop_duplicates(subset=['question'], keep='first')
    #import pdb; pdb.set_trace()
    print(f"Original len to process: {len(df)}")
    df = df[~df['question_id'].isin(processed_questions)]
    print(f"Reamining len to process: {len(df)}")
    
    if df.empty:
        print(f"Skipping {path_queries}, all questions processed.")
        checkpoint_df.to_parquet(os.path.join(args.path_save, path_queries))
        return
    
    llm_model = "qwen:32b"
    
    prompter = Prompter(
        model_type=llm_model, 
        ollama_host="http://kumo01.tsc.uc3m.es:11434")
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            results_4_unweighted = row[args.method_eval]
            flattened_list = [{'doc_id': entry['doc_id'], 'score': entry['score']} for subarray in results_4_unweighted for entry in subarray]
            top_docs = [el["doc_id"] for el in flattened_list][:top_k]
            
            for top_doc in top_docs:
            
                # ---------------------------------------------------------#
                # 3. ANSWER GENERATION
                #----------------------------------------------------------#
                with open(_3_INSTRUCTIONS_PATH, 'r') as file: template = file.read()
                
                ######################################
                # GENERATE ANSWER IN SOURCE LANGUAGE 
                ######################################
                passage_s = row.passage
                full_doc_s = row.full_doc
                
                formatted_template = template.format(question=row.question, passage=passage_s,full_document=(extend_to_full_sentence(full_doc_s, 100)+ " [...]"))
                
                answer_s, _ = prompter.prompt(question=formatted_template)
                print("Answer S:", answer_s)
                
                ######################################
                # GENERATE ANSWER IN TARGET LANGUAGE #
                ######################################
                passage_t = raw[raw.doc_id == top_doc].text.iloc[0]
                full_doc_t = raw[raw.doc_id == top_doc].full_doc.iloc[0]
                
                ##############################################
                # CHECK RELEVANCE OF PASSAGE TO THE QUESTION #
                ##############################################
                with open(RELEVANCE_PROMPT, 'r') as file: template = file.read()
                formatted_template = template.format(passage=passage_t, question=row.question)
                
                response, _ = prompter.prompt(question=formatted_template)
                relevance = 1 if "yes" in response.lower() else 0
                
                if relevance == 0:
                    answer_t = "I cannot answer the question given the context."
                else:
                    with open(_3_INSTRUCTIONS_PATH, 'r') as file: template = file.read()
                
                    formatted_template = template.format(question=row.question, passage=passage_t,full_document=(extend_to_full_sentence(full_doc_t, 100)+ " [...]"))
                    answer_t, _ = prompter.prompt(question=formatted_template)
                
                print("Answer T:", answer_t)
                
                if "cannot answer the question given the context" not in answer_t:
                    #-----------------------------------------------------#
                    # 4. DISCREPANCY DETECTION
                    # ------------------------------------------------------#
                    with open(_4_INSTRUCTIONS_PATH, 'r') as file: template = file.read()
                    
                    question = template.format(question=row.question, answer_1=answer_s, answer_2=answer_t)
                    
                    discrepancy, _ = prompter.prompt(question=question)
                    
                    label, reason = None, None
                    lines = discrepancy.splitlines()
                    for line in lines:
                        if line.startswith("DISCREPANCY_TYPE:"):
                            label = line.split("DISCREPANCY_TYPE:")[1].strip()
                        elif line.startswith("REASON:"):
                            reason = line.split("REASON:")[1].strip()
                    
            
                    if label is None or reason is None:
                        try:
                            discrepancy_split = discrepancy.split("\n")
                            reason = discrepancy_split[0].strip("\n").strip("REASON:").strip()
                            label = discrepancy_split[1].strip("\n").strip("DISCREPANCY_TYPE:").strip()
                        except:
                            label = discrepancy
                            reason = ""
                    print("Discrepancy:", label)
                    
                else:
                    if answer_t == "I cannot answer as the passage contains personal opinions.":
                        reason = "I cannot answer as the passage contains personal opinions."
                        label = "NOT_ENOUGH_INFO"
                    else:
                        reason = "I cannot answer given the context."
                        label = "NOT_ENOUGH_INFO"
                    
                    
                results.append({
                    "question_id": row.question_id,
                    "doc_id": top_doc,
                    "question": row.question,
                    "passage_s": passage_s,
                    "answer_s": answer_s,
                    "passage_t": passage_t,
                    "answer_t": answer_t,
                    "discrepancy": label,
                    "reason": reason
                })
                
                # Save checkpoint
                checkpoint_df = pd.DataFrame(results)
                checkpoint_df.to_parquet(checkpoint_file)
                #import pdb; pdb.set_trace()
        except Exception as e:
            print(f"Error with question {row.question_id}: {e}")
            continue
    
    # Save results
    results_df = pd.DataFrame(results)
    print(f"Generated new {len(results_df)} answers.")
    print(results_df.head())
    # concat with previous results
    results_df = pd.concat([results_df, checkpoint_df], ignore_index=True)
    print(f"Total answers: {results_df.shape[0]}")
    results_df.to_parquet(os.path.join(args.path_save, path_queries))
if __name__ == "__main__":
    main()