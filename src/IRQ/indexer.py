import logging
import pathlib
import os
import faiss
import time
import pandas as pd
import numpy as np
from scipy import sparse
from sentence_transformers import SentenceTransformer, util
from kneed import KneeLocator
from scipy.ndimage import uniform_filter1d


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
        thr = thr.lower()
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
        match:str
        Indicates whether to perform Exact or Approximate matches
    '''
    def __init__(self, file_path, model_path, model_name, thr, top_k,
                 match:str):
        super().__init__(file_path, model_path, model_name, thr, top_k)

        #Avoid possible errors
        match = match.lower()
        if match not in {'exact', 'approximate'}:
            raise(f'Invalid value for match: {match}, it has to be "exact" or "approximate"')
        else:
            self.match = match

        return
    
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
    
    def indexing_TBANN(self) -> None:
        embedding_size = 768  # Dimension of sentence embeddings
        min_clusters = 8  # Minimum number of clusters for small topic sizes
        top_k_hits = self.top_k  # Number of nearest neighbors to retrieve
        BATCH_SIZE = 32
        THR_TOPIC_ASSIGNMENT = self.thr #0.05
        top_k = self.top_k

        # Load SentenceTransformer model
        model = self.model

        LANGUAGE_GROUPS = {
            "EN": "EN",   # EN documents stay in the "EN" group
            "ES": "ES",   # ES documents stay in the "ES" group
            "T_EN": "ES", # Translated EN -> grouped with ES
            "T_ES": "EN", # Translated ES -> grouped with EN
        }


        # Paths
        PATH_SOURCE = self.file_path
        PATH_MODEL = self.model_path

        for LANG in ["EN", "ES"]:
            FAISS_SAVE_DIR = os.path.join(self.saving_path,f"indexes_{LANG}")

            # Ensure save directory exists
            os.makedirs(FAISS_SAVE_DIR, exist_ok=True)

            # Load data
            self._logger.info("-- Loading data...")
            raw = pd.read_parquet(PATH_SOURCE)

            thetas = sparse.load_npz(os.path.join(PATH_MODEL,"mallet_output",f"thetas_{LANG}.npz")).toarray()

            # Filter language documents
            topn = int(thetas.shape[1] / 3)
            #Filtering based on the translation changes and notation
            raw["lang_group"] = raw["id_preproc"].str.extract(r"(EN|ES|T_EN|T_ES)")[0].map(LANGUAGE_GROUPS)
            raw = raw[raw["lang_group"] == LANG].copy()

            #raw = raw[raw.id_preproc.str.contains(LANG)].copy()
            raw["thetas"] = list(thetas)
            raw["top_k"] = raw["thetas"].apply(lambda x: self.get_doc_top_tpcs(x, topn=topn))

            self._logger.info("-- Checking existing indices...")
            topic_indices = {}
            all_indices_exist = True 

            for topic in range(thetas.shape[1]):
                index_path = os.path.join(FAISS_SAVE_DIR , f"faiss_index_topic_{topic}.index")
                doc_ids_path = os.path.join(FAISS_SAVE_DIR , f"doc_ids_topic_{topic}.npy")

                if os.path.exists(index_path) and os.path.exists(doc_ids_path):
                    # Load the FAISS index and document IDs
                    self._logger.info(f"Loading indices for topic {topic}...")
                    index = faiss.read_index(str(index_path))
                    doc_ids = np.load(doc_ids_path, allow_pickle=True)
                    topic_indices[topic] = {"index": index, "doc_ids": doc_ids}
                else:
                    # If any index is missing, set flag to False
                    self._logger.info(f"Missing indices for topic {topic}")
                    all_indices_exist = False
                    break

            # We only generate embeddings and create new indices if any of the existing indices are missing
            if not all_indices_exist:
                self._logger.info("-- Generating embeddings...")
                corpus_embeddings = model.encode(
                    raw["raw_text"].tolist(), show_progress_bar=True, convert_to_numpy=True, batch_size=BATCH_SIZE
                )
                corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

                # Create FAISS indices for each topic
                for topic in range(thetas.shape[1]):
                    index_path = os.path.join(FAISS_SAVE_DIR, f"faiss_index_topic_{topic}.index")
                    doc_ids_path = os.path.join(FAISS_SAVE_DIR, f"doc_ids_topic_{topic}.npy")

                    if os.path.exists(index_path) and os.path.joins(doc_ids_path):
                        continue

                    self._logger.info(f"-- Creating index for topic {topic}...")
                    topic_embeddings = []
                    doc_ids = []

                    #TODO: Implement functionality to compute the adaptative threshold and use it here

                    for i, top_k in enumerate(raw["top_k"]):
                        for t, weight in top_k:
                            if t == topic and weight > self.thr:  # Relevance threshold for topic assignment
                                topic_embeddings.append(corpus_embeddings[i])
                                doc_ids.append(raw.iloc[i].doc_id)

                    if topic_embeddings:
                        topic_embeddings = np.array(topic_embeddings)
                        N = len(topic_embeddings)
                        n_clusters = max(int(4 * np.sqrt(N)), min_clusters)

                        self._logger.info(f"-- TOPIC {topic}: {N} documents, {n_clusters} clusters")

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
                self._logger.info("-- All indices are loaded.")
        return

    def indexing_ANN(self) -> None:
        BATCH_SIZE = 32
        embedding_size = 768  # Dimension of sentence embeddings
        # Load SentenceTransformer model
        model = self.model
        # Paths
        NR_TPCS = 30
        PATH_SOURCE = self.file_path
        PATH_MODEL = self.model_path
        
        
        FAISS_SAVE_DIR = os.path.join(self.saving_path,f"indexes_{LANG}")
        os.makedirs(FAISS_SAVE_DIR, exist_ok=True)

        LANGUAGE_GROUPS = {
            "EN": "EN",   # EN documents stay in the "EN" group
            "ES": "ES",   # ES documents stay in the "ES" group
            "T_EN": "ES", # Translated EN -> grouped with ES
            "T_ES": "EN", # Translated ES -> grouped with EN
        }

        for LANG in['EN', 'ES']:
            # Load data
            self._logger.info("-- Loading data...")
            raw = pd.read_parquet(PATH_SOURCE)
            thetas = sparse.load_npz(os.path.join(PATH_MODEL, "mallet_output", f"thetas_{LANG}.npz")).toarray()

            topn = int(thetas.shape[1] / 3)

            raw["lang_group"] = raw["id_preproc"].str.extract(r"(EN|ES|T_EN|T_ES)")[0].map(LANGUAGE_GROUPS)
            raw = raw[raw["lang_group"] == LANG].copy()

            raw["thetas"] = list(thetas)
            raw["top_k"] = raw["thetas"].apply(lambda x: self.get_doc_top_tpcs(x, topn=topn))

            '''
            thetas_en = sparse.load_npz(PATH_MODEL / "mallet_output" / f"thetas_EN.npz").toarray()
            raw_en["thetas"] = list(thetas_en)
            raw_en["top_k"] = raw_en["thetas"].apply(lambda x: self.get_doc_top_tpcs(x, topn=topn))
            '''

            # Generate embeddings if needed
            self._logger.info("-- Generating embeddings...")
            CORPUS_EMBEDDINGS_PATH = os.path.join(FAISS_SAVE_DIR, "corpus_embeddings.npy")

            if CORPUS_EMBEDDINGS_PATH.exists():
                self._logger.info("-- Loading existing corpus embeddings...")
                corpus_embeddings = np.load(CORPUS_EMBEDDINGS_PATH)
            else:
                self._logger.info("-- Generating embeddings...")
                corpus_embeddings = model.encode(
                    raw["text"].tolist(), show_progress_bar=True, convert_to_numpy=True, batch_size=BATCH_SIZE
                )
                corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
                np.save(CORPUS_EMBEDDINGS_PATH, corpus_embeddings)  # Save embeddings

            # ---- Create FAISS Index for Approximate Nearest Neighbors ----
            self._logger.info("-- Creating FAISS index for ANN...")
            FAISS_INDEX_PATH = os.path.join(FAISS_SAVE_DIR, "faiss_index_IVF.index")

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
        
            return
        
        def index(self) -> None:
            '''
            Selects and executes one of the two subroutines depending on
            the match type specified by the user
            '''
            if self.match == 'exact':
                pass

            return


class Retriever(NLPoperator):
    '''
    Does the retrieving of the documents upong recieving a query
    -------------
    Parameters:
        search_mode: either Topic based, exact or approximate
    '''
    def __init__(self, file_path, model_path, model_name, thr, top_k,
                 search_mode: str):
        super().__init__(file_path, model_path, model_name, thr, top_k)

        search_mode = search_mode.lower()
        if search_mode not in {'enn', 'ann', 'tb_enn', 'tb_ann'}:
            raise(f'Invalid value for match: {search_mode}, it has to be "enn", "ann" or "tb_enn", "tb_ann"')
        else:
            self.search_mode = search_mode

        self.indexer = Indexer(self.file_path,
                                self.model_path,
                                self.model_name,
                                self.thr,
                                self.top_k,
                                "exact")

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
    
    def topic_based_exact_search(self,query, theta_query, corpus_embeddings, raw, do_weighting=True):
        time_start = time.time()
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        results = []
        for topic, weight in theta_query:
            thr = self.thr[topic] if self.thr is not None else 0.05 #TODO: fix threshold assignment
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
            thr = self.thr[topic] if self.thr is not None else 0.05 #TODO: ASSIGN CORRECT VAL
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
