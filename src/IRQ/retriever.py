import logging
import pathlib
import os
import faiss
from tqdm import tqdm
import ast
import time
import pandas as pd
import numpy as np
from scipy import sparse, stats
import scikit_posthocs as sp
from sentence_transformers import SentenceTransformer, util
from kneed import KneeLocator
from tabulate import tabulate
from scipy.ndimage import uniform_filter1d
import re
from src.IRQ.indexer import NLPoperator

class Retriever(NLPoperator):
    '''
    Does the retrieving of the documents upong recieving a query
    -------------
    Parameters:
        search_mode: either Topic based, exact or approximate
    '''
    def __init__(self, file_path, model_path, model_name,
                 question_path, idx_config: dict = None):
        super().__init__(file_path, model_path, model_name)

        self.question_path = question_path
        #Default configuration for the index
        default_config = {
            "match": "ENN",
            "embedding_size": 768,
            "min_clusters": 8,
            "top_k_hits": 10,
            "batch_size": 32,
            "thr": '0.05',
            "top_k": 10,
            'storage_path': '/export/usuarios_ml4ds/ammesa/Data/4_indexed_data'
        }

        self.config = default_config if idx_config is None else {**default_config, **idx_config}

        self.storage_path = self.config['storage_path']

        self.output_df = None

        if self.config['match'] not in {'ENN', 'ANN', 'TB_ANN', 'TB_ENN'}:
            raise(f'Invalid value for match: {self.config['match']}, it has to be "ENN","ANN","TB_ENN","TB_ANN"')
        
                
        search_mode = self.config['match'].lower()
        if search_mode not in {'enn', 'ann', 'tb_enn', 'tb_ann'}:
            raise(f'Invalid value for match: {search_mode}, it has to be "enn", "ann" or "tb_enn", "tb_ann"')
        else:
            self.search_mode = search_mode

        self.indexer = Indexer(self.file_path,
                                self.model_path,
                                self.model_name,
                                self.config)
        
        self.thetas = None
        self.corpus_embeddings = None
        self.raw = None
        self.weight = False
        self.final_thrs = None
        self.path_mode = os.path.join(self.saving_path, self.search_mode.upper())
        self.queries = None
        return
    

    def read_thetas_em(self, bilingual: bool, topic_model:str) -> None:

        if bilingual:
            if topic_model == 'zeroshot':
                os.path.join(self.model_path, 'ZS_output', 'thetas.npy')
                thetas_path = os.path.join(self.model_path, 'ZS_output', 'thetas.npy')
                self.thetas = np.load(thetas_path)

                self._logger.info(f'Reading embeddings')

                path_em = os.path.join(self.indexer.saving_path,f'{self.config['match']}' ,'corpus_embeddings.npy')
                path_em = re.sub(r'([^/]+)/\1', r'\1', path_em)
                self.corpus_embeddings = np.load(path_em)

                self._logger.info(f'Reading raw docs')
                self.raw = pd.read_parquet(self.file_path)
            elif topic_model == 'mallet':
                thetas_path = os.path.join(self.model_path, 'mallet_output', f'thetas_EN.npz')
                thetas_en = sparse.load_npz(thetas_path).toarray()
                thetas_path = os.path.join(self.model_path, 'mallet_output', f'thetas_ES.npz')
                thetas_es = sparse.load_npz(thetas_path).toarray()
                self.thetas = np.vstack([thetas_en, thetas_es])

                self._logger.info(f'Reading embeddings')

                path_em = os.path.join(self.indexer.saving_path,f'{self.config['match']}' ,'corpus_embeddings.npy')
                path_em = re.sub(r'([^/]+)/\1', r'\1', path_em)
                self.corpus_embeddings = np.load(path_em)

                self._logger.info(f'Reading raw docs')
                self.raw = pd.read_parquet(self.file_path)

        else:
            lang = 'EN' 
            self._logger.info(f'Reading thetas of language {lang}')

            thetas_path = os.path.join(self.model_path,'mallet_output', 'EN',f'thetas.npz')
            thetas = sparse.load_npz(thetas_path).toarray()
            self.thetas = thetas

            self._logger.info(f'Reading embeddings')
            path_em = os.path.join(self.indexer.saving_path,f'{self.config['match']}' ,'corpus_embeddings.npy')
            path_em = re.sub(r'(\/[^\/]+)\1$', r'\1', path_em)
            self.corpus_embeddings = np.load(path_em)

            self._logger.info(f'Reading raw docs')
            self.raw = pd.read_parquet(self.file_path)


        return
    
    def exact_nearest_neighbors(self, query, corpus_embeddings, raw):
        #raw_en["thetas"] = list(thetas_en)
        #Is time necessary?
        time_start = time.time()
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        cosine_similarities = np.dot(corpus_embeddings, query_embedding.T).squeeze()
        top_k_indices = np.argsort(-cosine_similarities)[:self.config['top_k']]
        time_end = time.time()
        timelapse = time_end - time_start
        return [{"doc_id": raw.iloc[i].doc_id, "score": cosine_similarities[i]} for i in top_k_indices], timelapse
        
    def approximate_nearest_neighbors(self, query, faiss_index, doc_ids):
        time_start = time.time()
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        distances, indices = faiss_index.search(np.expand_dims(query_embedding, axis=0), self.config['top_k'])
        time_end = time.time()
        timelapse = time_end - time_start
        return [{"doc_id": doc_ids[idx], "score": dist} for dist, idx in zip(distances[0], indices[0]) if idx != -1], timelapse
    
    def topic_based_exact_search(self, query, theta_query, corpus_embeddings, raw, thr, do_weighting ):
        time_start = time.time()
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        results = []
        for topic, weight in theta_query:
            if thr is not None:
                thr = list(thr)
            thrs = thr[topic] if thr is not None else 0

            if weight > thrs:
                # Reset index so it matches corpus_embeddings indexing
                raw_reset_index = raw.reset_index(drop=True)
                topic_docs = raw_reset_index[raw_reset_index["top_k"].apply(lambda x: any(t == topic for t, _ in x))]
                
                # Now use `.iloc` to safely index into corpus_embeddings
                topic_embeddings = corpus_embeddings[topic_docs.index.to_numpy()]
                
                if len(topic_embeddings) == 0:
                    continue

                # Compute cosine similarity
                cosine_similarities = np.dot(topic_embeddings, query_embedding.T).squeeze()
                top_k_indices = np.argsort(-cosine_similarities)[:self.config['top_k']]

                for i in top_k_indices:
                    score = cosine_similarities[i] * weight if do_weighting else cosine_similarities[i]
                    results.append({"topic": topic, "doc_id": topic_docs.iloc[i].doc_id, "score": score})

        time_end = time.time()

        timelapse = time_end-time_start
        # Remove duplicates, keeping the highest score
        unique_results = {}
        for result in results:
            doc_id = result["doc_id"]
            if doc_id not in unique_results or result["score"] > unique_results[doc_id]["score"]:
                unique_results[doc_id] = result

        return sorted(unique_results.values(), key=lambda x: x["score"], reverse=True)[:self.config['top_k']], timelapse
    
    def topic_based_approximate_search(self, query, theta_query,thr,do_weighting):
        time_start = time.time()
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        results = []
        for topic, weight in theta_query:
            if thr is not None:
                thr = list(thr)
            thrs = thr[topic] if thr is not None else 0
            if weight > thrs:
                index_path = os.path.join(self.saving_path, f'TB_ANN/index_TB_ANN_{topic}' , f"faiss_index_topic_{topic}_EN.index")
                doc_ids_path = os.path.join(self.saving_path, f'TB_ANN/index_TB_ANN_{topic}' , f"doc_ids_topic_{topic}_EN.npy")
                if os.path.exists(index_path) and os.path.exists(doc_ids_path):
                    index = faiss.read_index(str(index_path))
                    doc_ids = np.load(doc_ids_path, allow_pickle=True)
                    distances, indices = index.search(np.expand_dims(query_embedding, axis=0), self.config['top_k'])
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
        timelapse = time_end-time_start

        return sorted(unique_results.values(), key=lambda x: x["score"], reverse=True)[:self.config['top_k']], timelapse
    
    def check_idx(self) -> bool:
        '''
        Checks whether the indexes for a given method
        have been computed or not, returns a boolean 
        depending on it.
        '''
        res = False

        self.search_mode = self.search_mode.upper()

        if self.search_mode == 'TB_ANN':
            suffix = 'index_TB_ANN_2/faiss_index_topic_2_EN.index'
        elif self.search_mode == 'TB_ENN':
            suffix = 'index_TB_ENN_2/faiss_index_topic_2_EN.index'
        elif self.search_mode == 'ANN':
            suffix = 'faiss_index_ANN_EN.index'
        else: 
            suffix = 'faiss_index_ENN_EN.index'

        if os.path.exists(os.path.join(self.path_mode, suffix)):
            res = True

        return res

    def retrieval_loop(self, bilingual: bool, n_tpcs : int, topic_model:str , weight : bool = False):
 
        self.weight = weight
         #Check if indexing has been done
        if not self.check_idx():
            self.indexer.index(bilingual=bilingual)
 
        #Get embeddings and thetas
        self.read_thetas_em(bilingual=bilingual, topic_model=topic_model)

        #Now iterate over the query stack 
        paths_ = os.listdir(self.question_path)

        for path_queries in paths_:
            LANG = 'EN'
            processed_rows = 0

            df_q = pd.read_excel(os.path.join(self.question_path, path_queries))
            #TODO: Solo para el entreno
            if not bilingual and "id" in df_q.columns:
                df_q = df_q.rename(columns={"id": "doc_id"})
            thetas = self.thetas
            lang_groups = {
                "EN": "EN",   
                "ES": "ES",   
                "T_EN": "ES", # Translated EN -> grouped with ES
                "T_ES": "EN", # Translated ES -> grouped with EN
            }
            #Needed parameter to adjust datasets in 2 langs
            self.raw['thetas'] = list(thetas)

            if bilingual:
                self.raw["lang_group"] = self.raw["id_preproc"].str.extract(r"(EN|ES|T_EN|T_ES)")[0].map(lang_groups)
                self.raw = self.raw[self.raw["lang_group"] == LANG].copy()
            self.raw['doc_id'] = self.raw['id_preproc']
            self.raw["top_k"] = self.raw["thetas"].apply(lambda x: self.get_doc_top_tpcs(x, topn=int(thetas.shape[1] / 3)))

            # Calculate threshold dynamically
            thrs_ = self.indexer.dynamic_thresholds(thetas, poly_degree=3, smoothing_window=5)
            if "llama" in path_queries:
                thrs_keep = [thrs_]
            else:
                thrs_keep = [None, thrs_]

            for thrs in thrs_keep:
                
                self._logger.info(f"Calculating results with thresholds: {thrs}")
                save_thr = "_dynamic" if thrs is not None else ""
                # initialize columns to store results
                for key_results in ["results"]:
                    df_q[key_results] = None
                for id_row, row in tqdm(df_q.iterrows(), total=df_q.shape[0]):
                    if n_tpcs != 30:
                        #doc_id = row.doc_id if isinstance(row.doc_id, str) else  

                        row[f"theta_{n_tpcs}"] = self.raw[self.raw.id_preproc == row.doc_id].thetas.values[0]
                        row[f"top_k_{n_tpcs}"] = self.raw[self.raw.id_preproc == row.doc_id].top_k.values[0]
                    
                    processed_rows += 1
                    print(100*processed_rows/len(df_q))
                    queries = ast.literal_eval(row.subqueries)
                    
                    if n_tpcs == 30:
                        theta_query = ast.literal_eval(row.top_k)
                    else:
                        theta_query = row[f"top_k_{n_tpcs}"]

                    results_1 = []
                    
                    time_1 = []

                    for query in queries:
                        if self.search_mode == 'ENN':
                            r1, t1 = self.exact_nearest_neighbors(query, self.corpus_embeddings, self.raw)
                        elif self.search_mode == 'ANN':
                            faiss_path = os.path.join(self.path_mode, 'faiss_index_ANN_EN.index')
                            faiss_index = faiss.read_index(str(faiss_path))
                            r1, t1 = self.approximate_nearest_neighbors(query, faiss_index, self.raw["doc_id"].tolist())
                        elif self.search_mode == 'TB_ENN':
                            r1, t1 =  self.topic_based_exact_search(query, theta_query, self.corpus_embeddings, self.raw, thrs, do_weighting=weight)
                        elif self.search_mode == 'TB_ANN':
                            r1, t1 = self.topic_based_approximate_search(query, theta_query, thrs, do_weighting=weight)

                        results_1.append(r1)

                        time_1.append(t1)
                        # print comparison of times
                        self._logger.info(f"{self.search_mode}: {t1:.2f}s")

                    
                    df_q.at[id_row, "results"] = results_1
                    df_q.at[id_row, "time"] = time_1

                if save_thr == '':
                    save_thr = 0
                path_save = os.path.join(self.storage_path, 'res')
                os.makedirs(path_save, exist_ok=True)
                df_q.to_parquet(os.path.join(path_save, path_queries.replace(".xlsx", f"_results_model_{n_tpcs}_tpc_{save_thr}_thr.parquet")))

                self._logger.info('Post-processing & saving')    
                # Convert all result lists to individual rows in one step
                columns_to_explode = ["results"]
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
                #Original: df_q_eval = df_q[['pass_id', 'doc_id', 'passage', 'top_k', 'question', 'queries', 'all_results']].copy()
                df_q_eval = df_q[['doc_id', 'full_doc', 'passage', 'question', 'subqueries', 'all_results', 'relevant_docs', 'time']].copy()

                
                df_q_eval = df_q_eval.explode("all_results", ignore_index=True)

                
                doc_map = self.raw.set_index("doc_id")["raw_text"].to_dict()
                df_q_eval["all_results_content"] = df_q_eval["all_results"].map(doc_map)
                

                #TODO:only necessary for TB_ANN and ANN of the pubmed dataset
                '''
                if self.config['match'] == 'TB_ANN':
                    id_mapping = dict(zip(df_q['Unnamed: 0'], df_q['doc_id']))
                    
                    # Apply transformation
                    df_q_eval['all_results'] = df_q_eval['all_results'].apply(lambda x: id_mapping.get(x, x))
  
                '''

                # Save the processed dataframe
                path_aux = os.path.join(self.storage_path, f'res',f'{self.search_mode}')
                os.makedirs(path_aux, exist_ok=True)
                path_save = os.path.join(path_aux , path_queries.replace(".xlsx", f"_results_model{n_tpcs}tpc_thr_{save_thr}_combined_to_retrieve_relevant.parquet"))
                df_q_eval.to_parquet(path_save)
                self.final_thrs = save_thr
                self.output_df = df_q_eval

                self.evaluation()

        return
    
    def precision_at_k(self, row, k):
        
        retrieved_docs = set(row[f"all_results"][:k])  
        relevant_docs = set(row["relevant_docs"])

        if not retrieved_docs:
            return 0.0

        return len(retrieved_docs & relevant_docs) / len(retrieved_docs)
    
    def recall_at_k(self, row, k):
        retrieved_docs = set(row[f"all_results"][:k])
        relevant_docs = set(row["relevant_docs"])

        if not relevant_docs:
            return 0.0

        return len(retrieved_docs & relevant_docs) / len(relevant_docs)
    
    def id_in_list(self, row, k):
        return 1 if row["relevant_docs"][0] in row["all_results"][:k] else 0
    
    def calibrated_hit_at_k(self, row, k):
        retrieved_docs = row[f"all_results"][:k] 
        relevant_doc = row["relevant_docs"][0]
        if relevant_doc in retrieved_docs:
            rank = retrieved_docs.index(relevant_doc) + 1  
            return 1 / rank  
        
        return 0  


    def multiple_mean_reciprocal_rank_at_k(self, row, k):
        retrieved_docs = row[f"all_results"][:k]
        relevant_docs = set(row["relevant_docs"])

        ranks = [i + 1 for i, doc in enumerate(retrieved_docs) if doc in relevant_docs]

        if not ranks:
            return 0.0

        result = np.mean([1 / rank for rank in ranks])
        
        '''
        ranks = []
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                ranks.append(i + 1)
        
        if not ranks:
            return 0

        numerator = np.mean(ranks)

        n = len(relevant_docs)
        denominator = (n / 2) + ((k + 1) * (n - len(ranks)))
        result = numerator / denominator
        '''
        return result
    
    def dcg_at_k(self, scores, k):

        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(scores[:k]))
    
    def ndcg_at_k(self, row, k):
        retrieved_docs = row[f"all_results"][:k]
        relevant_docs = set(row["relevant_docs"])
        
        # 1 if the document is relevant, 0 otherwise
        gains = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]
        
        dcg = self.dcg_at_k(gains, k)
        
        ideal_gains = sorted(gains, reverse=True)
        idcg = self.dcg_at_k(ideal_gains, k)
        
        return dcg / idcg if idcg > 0 else 0
    
       
    #TODO: Cambia en el futuro con 3,5,10
    def evaluation(self, ks:list = [3,5,10]) -> None:
        '''
        Outputs a table with the following metrics for the retrieval dataframe:
        Precision, Recall, MMRCK, DCG, NDCG
        '''
        df_aux = self.output_df
  
        metrics = [
            'mrr',
            'precision',
            'recall',
            'ndcg',
            'hit',
            'rank_hit'
        ]

        method_mapping = {
            "1": "ENN",
            "2": "ANN",
            "3_weighted": "TB-ENN-W",
            "3_unweighted": "TB-ENN",
            "4_weighted": "TB-ANN-W",
            "4_unweighted": "TB-ANN",
            "time_1": "ENN",
            "time_2": "ANN",
            "time_3_weighted": "TB-ENN-W",
            "time_3_unweighted": "TB-ENN",
            "time_4_weighted": "TB-ANN-W",
            "time_4_unweighted": "TB-ANN",
        }


        
        
        #import pdb; pdb.set_trace()

        df_aux['relevant_docs'] = df_aux['relevant_docs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        '''
        mapping_inv = dict(zip(og_df['id_preproc'], og_df['doc_id']))

        def map_relevant_docs_to_doc_ids(row):
            return [mapping_inv.get(id_preproc, None) for id_preproc in row]
        
        df_aux['relevant_docs'] = df_aux['relevant_docs'].apply(map_relevant_docs_to_doc_ids)
        '''
        #import pdb;pdb.set_trace()

        self.output_df = (
            df_aux.groupby('question')
            .agg({
                'doc_id': lambda x: list(x)[0],
                'full_doc': lambda x: list(x)[0],
                'passage': lambda x: list(x)[0],
                'subqueries': lambda x: list(x)[0],
                'all_results': lambda x: list(x),
                'relevant_docs': lambda x: list(x)[0],
                'all_results_content': lambda x: list(x),
                'time': lambda x: list(x)[0]
            })
            .reset_index()
        )

        for k in ks:
            self.output_df[f"mrr_{k}"] = self.output_df.apply(lambda x: self.multiple_mean_reciprocal_rank_at_k(x, k=k), axis=1)
            self.output_df[f"precision_{k}"] = self.output_df.apply(lambda x: self.precision_at_k(x,k=k), axis=1)
            self.output_df[f"recall_{k}"] = self.output_df.apply(lambda x: self.recall_at_k(x, k=k), axis=1)
            self.output_df[f"ndcg_{k}"] = self.output_df.apply(lambda x: self.ndcg_at_k(x, k=k), axis=1)
            self.output_df[f"hit_{k}"] = self.output_df.apply(lambda x: self.id_in_list(x, k=k), axis=1)
            self.output_df[f"rank_hit_{k}"] = self.output_df.apply(lambda x: self.calibrated_hit_at_k(x, k=k), axis=1)


        summary_data = []
        row = {'retrieval_method': self.search_mode}
        for k in ks:
            #row[f'avg_mrr@{k}_{self.search_mode}'] = self.output_df[f"mrr_{k}"].mean()
            row[f'avg_precision@{k}_{self.search_mode}'] = self.output_df[f"precision_{k}"].mean()
            row[f'avg_recall@{k}_{self.search_mode}'] = self.output_df[f"recall_{k}"].mean()
            row[f'avg_ndcg@{k}_{self.search_mode}'] = self.output_df[f"ndcg_{k}"].mean()
            row[f'avg_hit@{k}_{self.search_mode}'] = self.output_df[f"hit_{k}"].mean()
            row[f'avg_rank_hit@{k}_{self.search_mode}'] = self.output_df[f"rank_hit_{k}"].mean()

        row['thr'] = self.final_thrs
        row['weighted'] = self.weight
        summary_data.append(row)

        summary_table = pd.DataFrame(summary_data)
        summary_table = summary_table.round(4) 

        #Obtaining the data with full granularity:
        metric_cols = []
        for k in ks:
            metric_cols.extend([
                f"mrr_{k}",
                f"precision_{k}",
                f"recall_{k}",
                f"ndcg_{k}",
                f"hit_{k}",
                f"rank_hit_{k}",
            ])

        metric_cols.extend(['time'])

        metrics_only_df = self.output_df[metric_cols].copy()

        metrics_only_df["retrieval_method"] = self.search_mode
        metrics_only_df["thr"] = self.final_thrs
        metrics_only_df["weighted"] = self.weight
        
        csv_full_path = os.path.join(self.config['storage_path'],"full_metrics.parquet")

        if os.path.exists(csv_full_path):
            existing_df = pd.read_parquet(csv_full_path)
            combined_df = pd.concat([existing_df, metrics_only_df], ignore_index=True)
        else:
            combined_df = metrics_only_df
        combined_df['thr'] = combined_df['thr'].replace('_dynamic', 1)
        combined_df['thr'] = pd.to_numeric(combined_df['thr'], errors='coerce').astype('Int64')
        combined_df.to_parquet(csv_full_path, index=False)        

        csv_path = os.path.join(self.config['storage_path'],"metrics_retrieval.csv")

        write_header = not os.path.exists(csv_path)

        summary_table.to_csv(csv_path, mode='a', index=False, header=write_header)


        #TODO: Generar tabla para LaTEX
        print(tabulate(summary_table.head(10), headers='keys', tablefmt='github', showindex=False))

        #Confindence interval calculations
        ci_table = []
        for metric in metrics:
            for k in ks:
                col_name = f'{metric}_{k}'
                values = self.output_df[col_name]

                weighted_mean = np.average(values)
                weighted_var = np.average((values - weighted_mean) ** 2)
                weighted_std = np.sqrt(weighted_var)
                n = len(values)
                ci = 1.96 * (weighted_std / np.sqrt(n)) if n > 1 else 0

                ci_table.append((os.path.basename(self.question_path),
                                 self.config['match'],
                                 f'{metric}@{k}',
                                 weighted_mean,
                                 ci))
                        

        metric = 'time'
        values = self.output_df[metric].dropna()
        mean_values = values.mean()
        std_values = values.std()
        
        ci_table.append((os.path.basename(self.question_path),
                            self.config['match'],
                            metric,
                            mean_values,
                            std_values))
        df_weighted = pd.DataFrame(ci_table, columns=["FileID", "Method", "Metric", "Weighted Mean", "95% CI"])

        print(tabulate(df_weighted.head(10), headers='keys', tablefmt='github', showindex=False))
        csv_cin_path = os.path.join(self.config['storage_path'],"metrics_conf_ints.csv") 

        write_header = not os.path.exists(csv_cin_path)
        df_weighted.to_csv(csv_cin_path, mode='a', index=False, header=write_header)
        

        return