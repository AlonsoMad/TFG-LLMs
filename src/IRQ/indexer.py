import logging
import pathlib
import os
import faiss
from tqdm import tqdm
import ast
import time
import pandas as pd
import numpy as np
from scipy import sparse
from sentence_transformers import SentenceTransformer, util
from kneed import KneeLocator
from scipy.ndimage import uniform_filter1d
import datetime

class NLPoperator:
    '''
    Superclass for Indexer and Retriever
    defines the main paths and functions of I/O and log
    -------------
    Parameters:
        file_path: str
        Path to the dataframe to index

        model_path: str
        Path to the mallet directory, ends at "mallet_folder"

        saving_path: str
        Path to save the files

        model_name: str
        Name of the indexing/retrieving model

        thr: str
        Threshold used for relevance in topic assignment, if input is a number then it
        will be fixed, if it is the string "var" it will be dynamically computed.

        top_k: int
        Number of topics "K"
    '''
    def __init__(self,
                 file_path: str,
                 model_path: str,
                 model_name: str,
                 thr: str,
                 top_k: int):
        #TODO Add weighted for the search
        
        self.file_path = file_path
        self.model_path = model_path
        self.saving_path = '/export/usuarios_ml4ds/ammesa/Data/4_indexed_data'
        self.model_name = model_name
        self.top_k = top_k
        
        self.og_df = None
        self.thetas = None

        #Now to initialize the logger:
        logging.basicConfig(level='INFO')
        self._logger = logging.getLogger('NLPoperator')
        # Add a console handler to output logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Set handler level to INFO or lower
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        #Initialize the model
        self.model = SentenceTransformer(model_name)

        #Handlind the threshold
        
        thr = str(thr).lower()
        if thr == 'var':
            self.thr = 'var'
        elif float(thr):
            self.thr = float(thr)
        else:
            raise(f'Invalid value for thr: {thr}, it has to be a float or "var"')

        return
    
    def read_data(self) -> None: 
        '''
        Reads the specified paths for thetas and for the DataFrame
        initializes values to the og_df and thetas attributes
        '''
        if not os.path.exists(self.file_path):
            raise Exception('File path not found, check again')
        else:
            self.og_df = pd.read_parquet(self.file_path)
            self._logger.info("Dataframe read sucessfully!")
        return
    
    def get_doc_top_tpcs(self, doc_distr, topn=10):
        sorted_tpc_indices = np.argsort(doc_distr)[::-1]
        top = sorted_tpc_indices[:topn].tolist()
        return [(k, doc_distr[k]) for k in top if doc_distr[k] > 0]


class Indexer(NLPoperator):
    '''
    Class to perform indexation over the documents
    ------------
    Parameters:
        config:dict
        Dictionary of IRQ parameters, mainly match Topic Based or not, Exact
        or approximate; embedding size, batch size, threshold (inputed as a string
        as it could take the value "var"), top_k
    '''
    def __init__(self, file_path, model_path, model_name, thr, top_k,
                 config: dict = None):
        super().__init__(file_path, model_path, model_name, thr, top_k)

        default_config = {
            "match": "ENN",
            "embedding_size": 768,
            "min_clusters": 8,
            "top_k_hits": 10,
            "batch_size": 32,
            "thr": '0.01',
            "top_k": 10
        }

        self.config = default_config if config is None else {**default_config, **config}

        if self.config['match'] not in {'ENN', 'ANN', 'TB_ANN', 'TB_ENN'}:
            raise(f'Invalid value for match: {self.config['match']}, it has to be "ENN","ANN","TB_ENN","TB_ANN"')

        return
    
    def dynamic_thresholds(self, mat_, poly_degree=3, smoothing_window=5):
        '''
        Computes the threshold dynamically to obtain significant
        topics in the indexing phase.
        '''
        #TODO: Fix nonconvergence in variable
        thrs = []
        for k in range(len(mat_.T)):
            allvalues = np.sort(mat_[:, k].flatten())
            step = int(np.round(len(allvalues) / 1000))
            x_values = allvalues[::step]
            x_values = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
            y_values = (100 / len(allvalues)) * np.arange(0, len(allvalues))[::step]
            y_values_smooth = uniform_filter1d(y_values, size=smoothing_window)
            kneedle = KneeLocator(x_values, y_values_smooth, curve='concave', direction='increasing', interp_method='polynomial', polynomial_degree=poly_degree)
            thrs.append(kneedle.elbow)
        return thrs

    def query_faiss(self, question, theta_query, top_k=10):
        """
        Perform a weighted topic search using FAISS indices.
        """
        question_embedding = self.model.encode([question], normalize_embeddings=True)[0]
        results = []

        for topic, weight in theta_query:
            index_path = self.saving_path / f"faiss_index_topic_{topic}.index"
            doc_ids_path = self.saving_path / f"doc_ids_topic_{topic}.npy"

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

        
    def index(self) -> None:
        '''
        Generates the index for the dataset in file_path with
        the thetas generated by mallet and indexes Topic Based,
        Exact or Approximate matching it
        '''
        #Obtain parameters
        embedding_size = self.config['embedding_size']
        min_clusters = self.config['min_clusters']
        top_k_hits = self.config['top_k_hits']
        top_k = self.config['top_k']
        batch_size = self.config['batch_size']
        thr = self.config['thr']
        model = SentenceTransformer(self.model_name)

        lang_groups = {
            "EN": "EN",   
            "ES": "ES",   
            "T_EN": "ES", # Translated EN -> grouped with ES
            "T_ES": "EN", # Translated ES -> grouped with EN
        }

        path_source = self.file_path
        path_model = self.model_path
        path_save = os.path.join(self.saving_path, f'{self.config['match']}')
        os.makedirs(path_save, exist_ok=True)
        self.saving_path = path_save

        #Indexing loop and case switch
        for LANG in ['EN', 'ES']:
            self._logger.info(f'Loading data for the {LANG} dataset')
            #Obtain data and thetas
            raw = pd.read_parquet(path_source)
            thetas_path = os.path.join(path_model, 'mallet_output', f'thetas_{LANG}.npz')
            thetas = sparse.load_npz(thetas_path).toarray()

            raw["lang_group"] = raw["id_preproc"].str.extract(r"(EN|ES|T_EN|T_ES)")[0].map(lang_groups)
            raw = raw[raw["lang_group"] == LANG].copy()
            raw['thetas'] = list(thetas)
            raw["top_k"] = raw["thetas"].apply(lambda x: self.get_doc_top_tpcs(x, topn=int(thetas.shape[1] / 3)))

            #Obtain embeddings
            self._logger.info(f'Checking embeddings')
            embeddings_path = os.path.join(self.saving_path, 'corpus_embeddings.npy')

            if os.path.exists(embeddings_path):
                self._logger.info(f'Embeddings found! Loading')
                corpus_embeddings = np.load(embeddings_path)
            else:
                self._logger.info(f'Generating embeddings...')
                corpus_embeddings = model.encode(
                    raw["raw_text"].tolist(), show_progress_bar=True, convert_to_numpy=True, batch_size=batch_size
                )
                corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
                np.save(embeddings_path, corpus_embeddings)

            #Obtain threshold
            if thr == 'var':
                threshold = self.dynamic_thresholds(thetas)
            elif float(thr) or thr == '0.0':
                threshold = float(thr)
            else:
                raise(f'Invalid value for config threshold: {thr}, it has to be a float number in a str or "var"!')


            #Case switch with the different indexing modes:

            if self.config['match'] == 'ANN':
                n_clusters = 100 
                self._logger.info('Creating ANN indices')
                quantizer = faiss.IndexFlatIP(embedding_size)
                faiss_index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
                faiss_index.train(corpus_embeddings)
                faiss_index.add(corpus_embeddings)
                faiss.write_index(faiss_index,  os.path.join(self.saving_path, f"faiss_index_{self.config['match']}_{LANG}.index"))

            elif self.config['match'] == 'ENN':
                self._logger.info('Creating ENN indices')
                faiss_index = faiss.IndexFlatIP(embedding_size)
                faiss_index.add(corpus_embeddings)
                faiss.write_index(faiss_index,  os.path.join(self.saving_path, f"faiss_index_{self.config['match']}_{LANG}.index"))

            elif self.config['match'] in ['TB_ANN', 'TB_ENN']:
                topic_indices = {}

                for topic in range(thetas.shape[1]):
                    topic_path = os.path.join(self.saving_path, f'index_{self.config['match']}_{topic}')
                    os.makedirs(topic_path, exist_ok=True)
                    index_path = os.path.join(topic_path, f"faiss_index_topic_{topic}_{LANG}.index")
                    doc_ids_path = os.path.join(topic_path , f"doc_ids_topic_{topic}_{LANG}.npy")

                    if os.path.exists(index_path) and os.path.exists(doc_ids_path):
                        continue

                    self._logger.info(f"-- Creating index for topic {topic}...")
                    topic_embeddings = []
                    doc_ids = []

                    for i, top_k in enumerate(raw["top_k"]):
                        for t, weight in top_k:
                            if t == topic and weight > threshold:  # Relevance threshold for topic assignment
                                topic_embeddings.append(corpus_embeddings[i])
                                doc_ids.append(raw.iloc[i].doc_id)

                    if topic_embeddings:
                        self._logger.info(f'Starting indexing process for {self.config['match']}')
                        topic_embeddings = np.array(topic_embeddings)
                        N = len(topic_embeddings)
                        n_clusters = max(int(4 * np.sqrt(N)), min_clusters)

                        print(f"-- TOPIC {topic}: {N} documents, {n_clusters} clusters")

                        # Train IVF index
                        quantizer = faiss.IndexFlatIP(embedding_size)

                        index = (
                                 faiss.IndexIVFFlat(quantizer, embedding_size,n_clusters, faiss.METRIC_INNER_PRODUCT)
                                 if self.config['match'] == 'TB_ANN' else quantizer)
                        
                        if self.config['match'] == 'TB_ANN':
                            index.train(topic_embeddings)

                        index.add(topic_embeddings)

                        # Save the index and document IDs
                        faiss.write_index(index, index_path)
                        np.save(doc_ids_path, np.array(doc_ids))
                        topic_indices[topic] = {"index": index, "doc_ids": doc_ids}                


        return


class Retriever(NLPoperator):
    '''
    Does the retrieving of the documents upong recieving a query
    -------------
    Parameters:
        search_mode: either Topic based, exact or approximate
    '''
    def __init__(self, file_path, model_path, model_name, thr, top_k,
                 question_path, idx_config: dict = None):
        super().__init__(file_path, model_path, model_name, thr, top_k)

        self.question_path = question_path
        self.storage_path = '/export/usuarios_ml4ds/ammesa/Data/4_indexed_data'
        #Default configuration for the index
        default_config = {
            "match": "ENN",
            "embedding_size": 768,
            "min_clusters": 8,
            "top_k_hits": 10,
            "batch_size": 32,
            "thr": '0.01',
            "top_k": 10
        }

        self.config = default_config if idx_config is None else {**default_config, **idx_config}

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
                                str(self.thr),
                                self.top_k,
                                self.config)
        
        self.thetas = None
        self.corpus_embeddings = None
        self.raw = None
        self.path_mode = os.path.join(self.saving_path, self.search_mode.upper())
        self.queries = None
        return
    
    #TODO: Read query csv functionality
    def read_queries(self) -> None:
        
        return

    def read_thetas_em(self) -> None:
        lang = 'EN' #TODO: ask for help with the languagescd ..
        self._logger.info(f'Reading thetas of language {lang}')

        thetas_path = os.path.join(self.model_path,'mallet_output', f'thetas_{lang}.npz')
        thetas = sparse.load_npz(thetas_path).toarray()
        self.thetas = thetas

        self._logger.info(f'Reading embeddings')
        path_em = os.path.join(self.indexer.saving_path, self.search_mode.upper(), 'corpus_embeddings.npy')
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
        top_k_indices = np.argsort(-cosine_similarities)[:self.top_k]
        time_end = time.time()
        timelapse = time_end - time_start
        return [{"doc_id": raw.iloc[i].doc_id, "score": cosine_similarities[i]} for i in top_k_indices], timelapse
        
    def approximate_nearest_neighbors(self, query, faiss_index, doc_ids):
        time_start = time.time()
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        distances, indices = faiss_index.search(np.expand_dims(query_embedding, axis=0), self.top_k)
        time_end = time.time()
        timelapse = time_end - time_start
        return [{"doc_id": doc_ids[idx], "score": dist} for dist, idx in zip(distances[0], indices[0]) if idx != -1], timelapse
    
    def topic_based_exact_search(self, query, theta_query, corpus_embeddings, raw, do_weighting=True):
        time_start = time.time()
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        results = []
        for topic, weight in theta_query:
            thr = self.thr[topic] if self.thr is not None else self.thr #TODO: fix threshold assignment
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
                top_k_indices = np.argsort(-cosine_similarities)[:self.top_k]

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

        return sorted(unique_results.values(), key=lambda x: x["score"], reverse=True)[:self.top_k], timelapse
    
    def topic_based_approximate_search(self, query, theta_query, do_weighting=True):
        time_start = time.time()
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        results = []
        for topic, weight in theta_query:
            thr = self.thr[topic] if self.thr is not None else self.thr #TODO: ASSIGN CORRECT VAL
            if weight > thr:
                index_path = os.path.join(self.saving_path , f"faiss_index_topic_{topic}.index")
                doc_ids_path = os.path.join(self.saving_path, f"doc_ids_topic_{topic}.npy")
                if index_path.exists() and doc_ids_path.exists():
                    index = faiss.read_index(str(index_path))
                    doc_ids = np.load(doc_ids_path, allow_pickle=True)
                    distances, indices = index.search(np.expand_dims(query_embedding, axis=0), self.top_k)
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

        return sorted(unique_results.values(), key=lambda x: x["score"], reverse=True)[:self.top_k], timelapse
    
    def check_idx(self) -> bool:
        '''
        Checks whether the indexes for a given method
        have been computed or not, returns a boolean 
        depending on it.
        '''
        res = False

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

    def retrieval_loop(self, n_tpcs : int,  weight : bool = False):
        #Get embeddings and thetas
        self.read_thetas_em()

        #Check if indexing has been done
        if not self.check_idx():
            self.indexer.index()

        #Now iterate over the query stack 
        paths_ = os.listdir(self.question_path)

        for path_queries in paths_:
            LANG = 'EN'

            df_q = pd.read_excel(os.path.join(self.question_path, path_queries))

            thetas = self.thetas
            lang_groups = {
                "EN": "EN",   
                "ES": "ES",   
                "T_EN": "ES", # Translated EN -> grouped with ES
                "T_ES": "EN", # Translated ES -> grouped with EN
            }
            self.raw["lang_group"] = self.raw["id_preproc"].str.extract(r"(EN|ES|T_EN|T_ES)")[0].map(lang_groups)
            self.raw = self.raw[self.raw["lang_group"] == LANG].copy()
            self.raw['thetas'] = list(thetas)
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
                        row[f"theta_{n_tpcs}"] = self.raw[self.raw.id_preproc == row.doc_id].thetas.values[0]
                        row[f"top_k_{n_tpcs}"] = self.raw[self.raw.id_preproc == row.doc_id].top_k.values[0]

                    queries = ast.literal_eval(row.subqueries)
                    
                    if n_tpcs == 30:
                        theta_query = ast.literal_eval(row.top_k)
                    else:
                        theta_query = row[f"top_k_{n_tpcs}"]

                    results_1 = []
                    
                    time_1 = []

                    for query in queries:
                        
                        if self.search_mode == 'enn':
                            r1, t1 = self.exact_nearest_neighbors(query, self.corpus_embeddings, self.raw)
                        elif self.search_mode == 'ann':
                            faiss_path = os.path.join(self.path_mode, 'faiss_index_ANN_EN.index')
                            faiss_index = faiss.read_index(str(faiss_path))
                            r1, t1 = self.approximate_nearest_neighbors(query, faiss_index, self.raw["doc_id"].tolist(), self.top_k)
                        elif self.search_mode == 'tb_enn':
                            r1, t1 =  self.topic_based_exact_search(query, theta_query, self.corpus_embeddings, self.raw, self.top_k, thrs, do_weighting=weight)
                        else:
                            r1, t1 = self.topic_based_approximate_search(query, theta_query, self.top_k, thrs, do_weighting=True)

                        results_1.append(r1)

                        time_1.append(t1)

                        # print comparison of times
                        self._logger.info(f"{self.search_mode}: {t1:.2f}s")
                    
                    df_q.at[id_row, "results"] = results_1

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
                df_q_eval = df_q[['doc_id', 'full_doc', 'passage', 'question', 'subqueries', 'all_results']].copy()

                # Explode only once
                df_q_eval = df_q_eval.explode("all_results", ignore_index=True)

                # Use `map` instead of `apply` for better performance
                doc_map = self.raw.set_index("doc_id")["raw_text"].to_dict()
                df_q_eval["all_results_content"] = df_q_eval["all_results"].map(doc_map)

                # Save the processed dataframe
                path_aux = os.path.join(self.storage_path, f'res',f'{self.search_mode}')
                os.makedirs(path_aux, exist_ok=True)
                path_save = os.path.join(path_aux , path_queries.replace(".xlsx", f"_results_model{n_tpcs}tpc_thr_{save_thr}_combined_to_retrieve_relevant.parquet"))
                df_q_eval.to_parquet(path_save)

        return

class QueryEngine(NLPoperator):
    def __init__(self, file_path, model_path, model_name, thr, top_k):
        super().__init__(file_path, model_path, model_name, thr, top_k)
        return