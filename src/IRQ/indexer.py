import logging
import pathlib
import os
import faiss
from tqdm import tqdm
import ast
import time
from multiprocessing import get_context
import pandas as pd
import numpy as np
from scipy import sparse, stats
import scikit_posthocs as sp
from sentence_transformers import SentenceTransformer, util
from kneed import KneeLocator
from tabulate import tabulate
from scipy.ndimage import uniform_filter1d
import re
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    def __init__(
        self,
        file_path: str,
        model_path: str,
        model_name: str,
        saving_path : str = '/export/usuarios_ml4ds/ammesa/Data/4_indexed_data',
    ):
        self.file_path = file_path
        self.model_path = model_path
        self.saving_path = saving_path
        self.model_name = model_name
        
        self.og_df = None
        self.thetas = None

        # Now to initialize the logger:
        # ----------------------------------------------------------------------
        # TODO: Make a function to handle the logger
        # ----------------------------------------------------------------------
        logging.basicConfig(level='INFO')
        self._logger = logging.getLogger('NLPoperator')
        # Add a console handler to output logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Set handler level to INFO or lower
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        # ----------------------------------------------------------------------

        # Initialize the model
        self.model = SentenceTransformer(model_name)

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
            self._logger.info("Dataframe read successfully!")
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
    def __init__(
        self, 
        file_path, 
        model_path, 
        model_name, 
        config: dict = None
    ):
        super().__init__(file_path, model_path, model_name)

        # TODO: Move this to a configuration file (yaml / cfg)
        default_config = {
            "match": "ENN",
            "embedding_size": 384,
            "min_clusters": 8,
            "top_k_hits": 10,
            "batch_size": 32,
            "thr": '0.01',
            "top_k": 10
        }

        self.config = default_config if config is None else {**default_config, **config}

        if self.config['match'] not in {'ENN', 'ANN', 'TB_ANN', 'TB_ENN'}:
            raise(f'Invalid value for match: {self.config['match']}, it has to be "ENN","ANN","TB_ENN","TB_ANN"') #type: ignore

        return
    
    def dynamic_thresholds(self, mat_, poly_degree=3, smoothing_window=5):
        '''
        Computes the threshold dynamically to obtain significant
        topics in the indexing phase.
        '''
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

        
    def index(self, bilingual: bool, topic_model:str, nprobe:int=None) -> None:
        '''
        Generates the index for the dataset in file_path with
        the thetas generated by mallet and indexes Topic Based,
        Exact or Approximate matching it
        '''
        #Obtain parameters
        embedding_size = self.config['embedding_size']
        min_clusters = self.config['min_clusters']
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
        suffix = os.path.basename(self.model_path)
        path_save = os.path.join(self.saving_path, suffix, f'{self.config['match']}')
        os.makedirs(path_save, exist_ok=True)
        self.saving_path = path_save
        #Indexing loop and case switch
        for LANG in ['EN', 'ES']:
            self._logger.info(f'Loading data for the {LANG} dataset')
            #Obtain data and thetas
            raw = pd.read_parquet(path_source)
            #thetas_path = os.path.join(path_model, 'mallet_output', f'thetas_{LANG}.npz')
            if bilingual:
                if topic_model == 'zeroshot':   
                    thetas_path = os.path.join(path_model, 'ZS_output', 'thetas.npy')
                    thetas_path = re.sub(r'([^/]+)/(?:\1)(/|$)', r'\1\2', thetas_path)
                    thetas = np.load(thetas_path)

                    raw['thetas'] = list(thetas)
                    raw["lang_group"] = raw["id_preproc"].str.extract(r"(EN|ES|T_EN|T_ES)")[0].map(lang_groups)
                    #raw = raw[raw["lang_group"] == LANG].copy()

                elif topic_model == 'mallet':
                    thetas_path = os.path.join(path_model, 'mallet_output', f'thetas_{LANG}.npz')
                    thetas_path = re.sub(r'([^/]+)/\1', r'\1', thetas_path)
                    thetas = sparse.load_npz(thetas_path).toarray()

                    raw["lang_group"] = raw["id_preproc"].str.extract(r"(EN|ES|T_EN|T_ES)")[0].map(lang_groups)
                    raw = raw[raw["lang_group"] == LANG].copy()
                    raw['thetas'] = list(thetas)
            else:
                if topic_model == 'lda':
                #thetas_path = os.path.join(path_model, 'mallet_output', self.config['lang1'] ,'thetas.npz')
                    thetas_path = os.path.join(path_model, f'n_topics_{self.config['k']}','mallet_output', self.config['lang1'] ,'thetas.npz')
                    thetas = sparse.load_npz(thetas_path).toarray()
                else: 
                    thetas_path = os.path.join(path_model, f'n_topics_{self.config['k']}','ZS_output','thetas.npy')
                    thetas = np.load(thetas_path)


                raw['thetas'] = list(thetas)

            # if bilingual:
            #     raw["lang_group"] = raw["id_preproc"].str.extract(r"(EN|ES|T_EN|T_ES)")[0].map(lang_groups)
            #     raw = raw[raw["lang_group"] == LANG].copy()
                
            raw["top_k"] = raw["thetas"].apply(lambda x: self.get_doc_top_tpcs(x, topn=int(thetas.shape[1] / 3)))

            #Obtain embeddings
            self._logger.info(f'Checking embeddings')
            embeddings_path = os.path.join(self.saving_path, 'corpus_embeddings.npy')

            if 'doc_id' not in raw.columns:
                if 'id_preproc' not in raw.columns:
                    raw['doc_id'] = raw.index
                else:
                    raw['doc_id'] = raw['id_preproc']

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
                threshold = threshold = np.full(thetas.shape[1], float(thr))
            else:
                raise(f'Invalid value for config threshold: {thr}, it has to be a float number in a str or "var"!') # type: ignore


            #Case switch with the different indexing modes:

            if self.config['match'] == 'ANN':
                n_clusters = 100 
                self._logger.info('Creating ANN indices')
                quantizer = faiss.IndexFlatIP(embedding_size)
                faiss_index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
                faiss_index.train(corpus_embeddings)
                if nprobe is not None:
                    faiss_index.nprobe = nprobe
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

                    thrs = threshold[topic] if thr is not None else 0

                    for i, top_k in enumerate(raw["top_k"]):
                        for t, weight in top_k:
                            if t == topic and weight > thrs:  # Relevance threshold for topic assignment
                                topic_embeddings.append(corpus_embeddings[i])
                                doc_ids.append(raw.iloc[i].doc_id)
                                

                    if topic_embeddings:
                        self._logger.info(f'Starting indexing process for {self.config['match']}')
                        #import pdb; pdb.set_trace()
                        topic_embeddings = np.asarray(topic_embeddings)
                        #topic_embeddings = topic_embeddings[:, np.newaxis]
                        
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

                        #topic_embeddings = np.array(topic_embeddings)
                        self._logger.info(f"Shape of topic_embeddings: {topic_embeddings.shape}")

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
    def __init__(self, file_path, model_path, model_name,
                 question_path, idx_config: dict = None):
        super().__init__(file_path, model_path, model_name)

        self.question_path = question_path
        #Default configuration for the index
        default_config = {
            "match": "TB_ENN",
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
            raise(f'Invalid value for match: {self.config['match']}, it has to be "ENN","ANN","TB_ENN","TB_ANN"') # type: ignore
        
                
        search_mode = self.config['match'].lower()
        if search_mode not in {'enn', 'ann', 'tb_enn', 'tb_ann'}:
            raise(f'Invalid value for match: {search_mode}, it has to be "enn", "ann" or "tb_enn", "tb_ann"') # type: ignore
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
        self.path_mode = os.path.join(self.storage_path, self.search_mode)
        self.queries = None
        return
    
    def update_q_path(self, path:str):
        self.question_path = path
        return
    

    def read_thetas_em(self, bilingual: bool, topic_model:str) -> None:

        if bilingual:
            if topic_model == 'zeroshot':
                os.path.join(self.model_path, 'ZS_output', 'thetas.npy')
                thetas_path = os.path.join(self.model_path, 'ZS_output', 'thetas.npy')
                thetas_path = re.sub(r'([^/]+)/\1', r'\1', thetas_path)

                self.thetas = np.load(thetas_path)

                self._logger.info(f'Reading embeddings')

                path_em = os.path.join(self.indexer.saving_path,f'{self.config['match']}' ,'corpus_embeddings.npy')
                path_em = re.sub(r'([^/]+)/\1', r'\1', path_em)
                self.corpus_embeddings = np.load(path_em)

                self._logger.info(f'Reading raw docs')
                self.raw = pd.read_parquet(self.file_path)
                
            elif topic_model == 'mallet':
                thetas_path = os.path.join(self.model_path, 'mallet_output', f'thetas_EN.npz')
                thetas_path = re.sub(r'([^/]+)/\1', r'\1', thetas_path)

                thetas_en = sparse.load_npz(thetas_path).toarray()
                thetas_path = os.path.join(self.model_path, 'mallet_output', f'thetas_ES.npz')
                thetas_path = re.sub(r'([^/]+)/\1', r'\1', thetas_path)

                thetas_es = sparse.load_npz(thetas_path).toarray()
                self.thetas = np.vstack([thetas_en, thetas_es])

                self._logger.info(f'Reading embeddings')

                path_em = os.path.join(self.indexer.saving_path,f'{self.config['match']}' ,'corpus_embeddings.npy')
                path_em = re.sub(r'([^/]+)/\1', r'\1', path_em)
                self.corpus_embeddings = np.load(path_em)

                self._logger.info(f'Reading raw docs')
                self.raw = pd.read_parquet(self.file_path)

        else:
            if topic_model == 'lda':
                lang = self.config['lang1'] 
                self._logger.info(f'Reading thetas of language {lang}')

                thetas_path = os.path.join(self.model_path,f'n_topics_{self.config['k']}','mallet_output', self.config['lang1'],f'thetas.npz')
                thetas = sparse.load_npz(thetas_path).toarray()
                self.thetas = thetas

                self._logger.info(f'Reading embeddings')

                suffix = os.path.basename(self.model_path)

                path_em = os.path.join(self.indexer.saving_path,suffix,f'{self.config['match']}' ,'corpus_embeddings.npy')
                path_em = re.sub(r'(\/[^\/]+)\1$', r'\1', path_em)
                path_em = re.sub(r'(\/[^\/]+\/[^\/]+)(\1)', r'\1', path_em)
                self.corpus_embeddings = np.load(path_em)

                self._logger.info(f'Reading raw docs')
                self.raw = pd.read_parquet(self.file_path)

                if 'doc_id' not in self.raw.columns:
                    if 'id_preproc' not in self.raw.columns:
                        self.raw['doc_id'] = self.raw.index
                    else:
                        self.raw['doc_id'] = self.raw['id_preproc']
            elif topic_model == 'zeroshot':
                
                thetas_path = os.path.join(self.model_path,f'n_topics_{self.config['k']}','scoutput','thetas.npy')
                self.thetas = np.load(thetas_path)

                self._logger.info(f'Reading embeddings')

                path_em = os.path.join(self.indexer.saving_path,f'{self.config['match']}' ,'corpus_embeddings.npy')
                path_em = re.sub(r'([^/]+)/\1', r'\1', path_em)
                self.corpus_embeddings = np.load(path_em)

                self._logger.info(f'Reading raw docs')
                self.raw = pd.read_parquet(self.file_path)
            
        return
    
    def exact_nearest_neighbors(self, query, faiss_index, corpus_embeddings, raw):
        #raw_en["thetas"] = list(thetas_en)
        # TODO: No utilizas los indices de faiss. Ya que indexas, usa el índice
        time_start = time.time()
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        distances, indices = faiss_index.search(np.expand_dims(query_embedding, axis=0), self.config['top_k'])
        time_end = time.time()
        timelapse = time_end - time_start
        return [{"doc_id": raw.iloc[i].doc_id, "score": distances} for distances, i in zip(distances[0], indices[0]) if i != -1], timelapse
        
    def approximate_nearest_neighbors(self, query, faiss_index, doc_ids):
        time_start = time.time()
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        distances, indices = faiss_index.search(np.expand_dims(query_embedding, axis=0), self.config['top_k'])
        time_end = time.time()
        timelapse = time_end - time_start
        return [{"doc_id": doc_ids[idx], "score": dist} for dist, idx in zip(distances[0], indices[0]) if idx != -1], timelapse
    
    def topic_based_exact_search(self, query, theta_query, corpus_embeddings, raw, thr, do_weighting):
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
        suffix = os.path.basename(self.model_path)
        path_save = os.path.join(self.saving_path, suffix)
        for topic, weight in theta_query:
            if thr is not None:
                thr = list(thr)
            thrs = thr[topic] if thr is not None else 0
            if weight > thrs:
                index_path = os.path.join(path_save, f'TB_ANN/index_TB_ANN_{topic}' , f"faiss_index_topic_{topic}_EN.index")
                doc_ids_path = os.path.join(path_save, f'TB_ANN/index_TB_ANN_{topic}' , f"doc_ids_topic_{topic}_EN.npy")
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
            suffix = 'non_existent_directory_to_force_reindexing'#'faiss_index_ANN_EN.index'
        else: 
            suffix = 'faiss_index_ENN_EN.index'
            self.nprobe=None

        
        if os.path.exists(os.path.join(self.path_mode, suffix)):
            res = True

        return res

    def process_single_row(self, id_row, row, n_tpcs, thrs, weight):
        """Process a single row from the dataframe."""
        results = []
        query_times = []
        
        # Debug output
        self._logger.debug(f"Processing row {id_row}")
        
        # Handle theta and top_k based on n_tpcs
        # Use class attribute self.raw instead of passing raw_data
        matching_rows = self.raw[self.raw.doc_id == row.doc_id]
        if len(matching_rows) > 0:
            row_theta = matching_rows.thetas.values[0]
            row_top_k = matching_rows.top_k.values[0]
        else:
            self._logger.warning(f"No matching doc_id found for {row.doc_id}")
            return id_row, [], 0, None, None
        
        # Get queries from row
        #queries = row.subqueries  # Assuming subqueries is already a list
        
        # Get theta_query based on n_tpcs

        theta_query = row_top_k
        query_parsed = self.clean_and_parse_subq(row['subqueries'])
        
        # Process each query
        for query in query_parsed:
            
            if self.search_mode == 'ENN':
                base_name = os.path.basename(self.path_mode)
                if base_name.lower() == "enn":
                    path = os.path.join(os.path.dirname(self.path_mode), "ENN")
                faiss_path = os.path.join(path, 'faiss_index_ENN_EN.index')
                faiss_index = faiss.read_index(str(faiss_path))
                r1, t1 = self.exact_nearest_neighbors(query, faiss_index,self.corpus_embeddings, raw=self.raw)
            elif self.search_mode == 'ANN':
                base_name = os.path.basename(self.path_mode)
                if base_name.lower() == "ann":
                    path = os.path.join(os.path.dirname(self.path_mode), "ANN")
                faiss_path = os.path.join(path, 'faiss_index_ANN_EN.index')
                faiss_index = faiss.read_index(str(faiss_path))
                r1, t1 = self.approximate_nearest_neighbors(query, faiss_index, self.raw["doc_id"].tolist())
            elif self.search_mode == 'TB_ENN':
                r1, t1 = self.topic_based_exact_search(query, theta_query, self.corpus_embeddings, self.raw, thrs, do_weighting=weight)
            elif self.search_mode == 'TB_ANN':
                r1, t1 = self.topic_based_approximate_search(query, theta_query, thrs, do_weighting=weight)
            else:
                r1, t1 = None, 0
                
            results.append(r1)
            query_times.append(t1)
            
            # Log time
            self._logger.info(f"{self.search_mode}: {t1:.2f}s")
        
        theta_value = row_theta if n_tpcs != 30 else None
        top_k_value = row_top_k if n_tpcs != 30 else None
        
        return id_row, results, np.average(query_times), theta_value, top_k_value

    def parallelized_search(self, df_q, n_tpcs=30, thrs=0, weight=False, save_thr='', path_queries=''):
        """Thread-based parallelized version of the search function."""
        start_time = time.time()
        self._logger.info(f"Starting parallelized search with {n_tpcs} topics using threads...")
        
        # Make a copy of the dataframe to avoid race conditions
        df_q_copy = df_q.copy()
        
        # Get total number of rows for progress tracking
        total_rows = len(df_q_copy)
        
        # Determine number of threads to use (adjust based on your specific workload)
        # For I/O bound tasks, you can use more threads than CPU cores
        max_workers = min(32, total_rows, os.cpu_count() / 2)
        self._logger.info(f"Using ThreadPoolExecutor with {max_workers} workers")
        
        # Create a progress bar for visual feedback
        pbar = tqdm(total=total_rows, desc="Processing rows")
        
        # Using ThreadPoolExecutor instead of ProcessPool to avoid serialization issues
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_to_row = {
                executor.submit(self.process_single_row, id_row, row, n_tpcs, thrs, weight): (id_row, row)
                for id_row, row in df_q_copy.iterrows()
            }
            
            # Process results as they complete
            for future in as_completed(future_to_row):
                try:
                    id_row, results, avg_time, row_theta, row_top_k = future.result()
                    
                    # Update results in the original dataframe
                    df_q.at[id_row, "results"] = results
                    df_q.at[id_row, "time"] = avg_time
                    
                    # Update theta and top_k if n_tpcs != 30
                    if n_tpcs != 30 and row_theta is not None and row_top_k is not None:
                        df_q.at[id_row, f"theta_{n_tpcs}"] = row_theta
                        df_q.at[id_row, f"top_k_{n_tpcs}"] = row_top_k
                    
                    # Update progress bar
                    pbar.update(1)
                    
                except Exception as exc:
                    # Get the row information for debugging
                    id_row, row = future_to_row[future]
                    self._logger.error(f"Row {id_row} generated an exception: {exc}")
                    # Continue processing other rows
        
        # Close progress bar
        pbar.close()
        
        # Save results
        if save_thr == '':
            save_thr = 0
        path_save = os.path.join(self.storage_path, 'res')
        os.makedirs(path_save, exist_ok=True)
        
        result_file = os.path.join(path_save, path_queries.replace(".xlsx", f"_results_model_{n_tpcs}_tpc_{save_thr}_thr.parquet"))
        df_q.to_parquet(result_file)
        
        self._logger.info('Post-processing & saving')
        
        # Convert all result lists to individual rows in one step
        columns_to_explode = ["results"]
        df_q = df_q.explode(columns_to_explode, ignore_index=True)
        
        total_time = time.time() - start_time
        self._logger.info(f"Total processing time: {total_time:.2f}s")
        
        return df_q

    def clean_and_parse_subq(self, subquery_str):
        if isinstance(subquery_str, str):
            try:
                fixed_str = subquery_str.replace('\n', ',')
                subquery_list = ast.literal_eval(fixed_str)
                return [s.strip() for s in subquery_list]
            except Exception as e:
                print(f"Error parsing: {subquery_str}\nError: {e}")
                return subquery_str
        else:
            return subquery_str
        

    def retrieval_loop(self, bilingual: bool, n_tpcs : int, topic_model:str , weight : bool = False,
                       evaluation_mode:bool=False, question_df: pd.DataFrame = None, nprobe:int=None,
                       parallel:bool=True):
        self._logger.info('Entering the retrieval loop!')
        self.nprobe = nprobe
        self.weight = weight
         #Check if indexing has been done
        if not self.check_idx():
            self.indexer.index(bilingual=bilingual, topic_model=topic_model, nprobe=nprobe)
 
        #Get embeddings and thetas
        self.read_thetas_em(bilingual=bilingual, topic_model=topic_model)

        #Now iterate over the query stack 
        paths_ = os.listdir(self.question_path)

        for path_queries in paths_:
            LANG = 'EN'
            processed_rows = 0

            if question_df is not None:
                df_q = question_df
            else:
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
            #import pdb;pdb.set_trace()
            if bilingual:
                self.raw["lang_group"] = self.raw["id_preproc"].str.extract(r"(EN|ES|T_EN|T_ES)")[0].map(lang_groups)
                self.raw_lang = self.raw[self.raw["lang_group"] == LANG].copy()
                
                #raw_o_lang just means the raw files in the OTHER language
                self.raw_o_lang = self.raw[self.raw["lang_group"] != LANG].copy()
                #self.raw = self.raw_lang

            #self.raw['doc_id'] = self.raw['id_preproc']
            self.raw["top_k"] = self.raw["thetas"].apply(lambda x: self.get_doc_top_tpcs(x, topn=int(thetas.shape[1] / 3)))

            if 'doc_id' not in self.raw.columns:
                self.raw['doc_id'] = self.raw['id_preproc']
            
            
            if 'raw_text' not in df_q.columns:
                df_q['raw_text'] = df_q['full_doc']
            
            if (len(df_q) != len(self.raw)) and not bilingual:
                df_q = df_q[df_q['doc_id'].isin(self.raw['doc_id'])]

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
                
                
                if parallel:
                    df_q = self.parallelized_search(
                        df_q, 
                        n_tpcs=n_tpcs, 
                        thrs=thrs, 
                        weight=weight, 
                        save_thr=save_thr, 
                        path_queries=path_queries
                    )
                else:
                    for id_row, row in tqdm(df_q.iterrows(), total=df_q.shape[0]):
                        #if n_tpcs != 30:
                            
                        row[f"theta_{n_tpcs}"] = self.raw[self.raw.doc_id == row.doc_id].thetas.values[0]
                        row[f"top_k_{n_tpcs}"] = self.raw[self.raw.doc_id == row.doc_id].top_k.values[0]
                        
                        queries = row.subqueries#queries = ast.literal_eval(row.subqueries)
                        #if n_tpcs == 30:
                        #   theta_query = ast.literal_eval(row[f"top_k_{n_tpcs}"])
                        #else:
                        theta_query = row[f"top_k_{n_tpcs}"]

                        results_1 = []
                        
                        time_1 = []

                        
                        query_parsed = self.clean_and_parse_subq(row['subqueries'])

                        for query in query_parsed:

                            if self.search_mode == 'ENN':
                                base_name = os.path.basename(self.path_mode)
                                if base_name.lower() == "enn":
                                    path = os.path.join(os.path.dirname(self.path_mode), "ENN")
                                faiss_path = os.path.join(path, 'faiss_index_ENN_EN.index')
                                faiss_index = faiss.read_index(str(faiss_path))
                                r1, t1 = self.exact_nearest_neighbors(query, faiss_index,self.corpus_embeddings, raw=self.raw)
                                
                            elif self.search_mode == 'ANN':
                                base_name = os.path.basename(self.path_mode)
                                if base_name.lower() == "ann":
                                    path = os.path.join(os.path.dirname(self.path_mode), "ANN")
                                faiss_path = os.path.join(path, 'faiss_index_ANN_EN.index')
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

                        df_q.at[id_row, "time"] = np.average(time_1)
                
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
                        elif isinstance(content, dict):
                            doc_ids.add(content['doc_id'])
                    return list(doc_ids)

                df_q["all_results"] = df_q[columns_to_explode].apply(combine_results, axis=1)

                # Select only necessary columns
                #Original: df_q_eval = df_q[['pass_id', 'doc_id', 'passage', 'top_k', 'question', 'queries', 'all_results']].copy()
                df_q_eval = df_q[['doc_id', 'raw_text', 'question', 'subqueries', 'all_results', 'relevant_docs', 'time']].copy()

                
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

                #Puedo comentarlo para funcionalidad final
                if evaluation_mode:
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
        return 1 if set(row["relevant_docs"]) in row["all_results"][:k] else 0
    
    def calibrated_hit_at_k(self, row, k):
        retrieved_docs = row[f"all_results"][:k]
        relevant_doc = set(row["relevant_docs"])

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

        df_aux['relevant_docs'] = df_aux['relevant_docs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df_aux['all_results'] = df_aux['all_results'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df_aux['all_results'] = df_aux['all_results'].apply(lambda x: int(x) if isinstance(x, (np.integer, float)) and not pd.isna(x) else x)


        self.output_df = (
            df_aux.groupby('question')
            .agg({
                'doc_id': lambda x: list(x)[0],
                'raw_text': lambda x: list(x)[0],
                'subqueries': lambda x: list(x)[0],
                'all_results': lambda x: list(x),
                'relevant_docs': lambda x: list(x),
                'all_results_content': lambda x: list(x),
                'time': lambda x: list(x)[0]
            })
            .reset_index()
        )

        relevant_docs = self.output_df.get('relevant_docs')


        #ZAPATA ASQUEROSA ELIMINAR SI POSIBLE
        if (
            self.config['match'] == 'TB_ANN' and
            relevant_docs is not None and
            len(relevant_docs) > 0 and
            isinstance(relevant_docs[0], list) and
            len(relevant_docs[0]) > 0 and
            isinstance(relevant_docs[0][0], str)
        ):
            og_df = pd.read_parquet(self.file_path)

            mapping_inv = dict(zip(og_df['id_preproc'], og_df['doc_id']))

            def map_relevant_docs_to_doc_ids(row):
                return [mapping_inv.get(id_preproc) for id_preproc in row]

            self.output_df['relevant_docs'] = self.output_df['relevant_docs'].apply(map_relevant_docs_to_doc_ids)
           

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
        if self.nprobe is None:
            metrics_only_df["nprobe"] = 'None'
        else:
            metrics_only_df["nprobe"] = self.nprobe
            
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