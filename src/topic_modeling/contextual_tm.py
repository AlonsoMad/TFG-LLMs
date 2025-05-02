'''
Wrapper class for the contextualized-topic-models package. 
Available here: https://github.com/MilaNLProc/contextualized-topic-models.git
--------------
Gets as an input a preprocessed corpus with 2 language varieties and produces:

'''
import logging
import os
import json
import pandas as pd
import numpy as np
from src.CTM.contextualized_topic_models.models.ctm import ZeroShotTM
from src.CTM.contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation

class ContextualTM(object):
    
    def __init__(self,
                 input_path: str,
                 output_path: str,
                 input_f_name: str,
                 lang1: str,
                 lang2: str,
                 df1: pd.DataFrame = None,
                 df2: pd.DataFrame = None,
                 corpus_og: list = None,
                 corpus_pre: list = None,
                 logger: logging.Logger = None
                ) -> None:
        
        self.input_path = input_path
        self.output_path = output_path
        self.input_f_name = input_f_name

        self.corpus_path = 'corpus'
        self.ZS_output_path = 'ZS_output'

        self.lang1 = lang1
        self.lang2 = lang2

        #Training-relevant parameters
        self.preparer = None
        self.model = None
        self.training_dataset = None
        self.test_dataset = None
        #borrar estos datos manuales
        self.thetas = None
        self.betas = None
        self.topics = None
        self.vocab = None

        #Usar la función built-in directamente
        self.ldavis_data = None

        #Create the empty lists for the corpuses
        self.corpus_og = corpus_og if corpus_og is not None else list()
        self.corpus_pre = corpus_pre if corpus_pre is not None else list()

        #TODO: Might be sucbject to changes due to preproc-id
        self.df1 = df1 if df1 is not None else pd.DataFrame(columns=['id',
                                                                    'raw_text',
                                                                    'lemmas',
                                                                    'lang',
                                                                    'doc_id'])
        self.df2 = df2 if df2 is not None else pd.DataFrame(columns=['id',
                                                                    'raw_text',
                                                                    'lemmas',
                                                                    'lang',
                                                                    'doc_id'])
        
        if logger:
            self._logger = logger
        else:
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('Segmenter')
            # Add a console handler to output logs to the console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)  # Set handler level to INFO or lower
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self._logger.addHandler(console_handler)

        
        return
        
    def read_dataframes(self) -> None:
        '''
        Checks the in_directory path and
        saves the file as a PD dataframe (input_df)
        '''
        if not os.path.exists(self.input_path):
          raise Exception('Path not found, check again')

        elif not os.path.isfile(os.path.join(self.input_path, self.input_f_name)):
          raise Exception('File not found, check again')

        else:
          self.df1 = pd.read_parquet(os.path.join(self.input_path, self.input_f_name))
          self._logger.info("File read sucessfully!")

        return
    
    def prepare_corpus(self) -> None:
        '''
        Extracts the relevant info out of the dataframe and produces 
        a corpus of the raw text and one of the lemmas.
        '''
        self.corpus_og = self.df1['raw_text'].tolist()
        self.corpus_pre = self.df1['lemmas'].tolist()

        self.preparer = TopicModelDataPreparation("paraphrase-distilroberta-base-v2")
        
        self.training_dataset = self.preparer.fit(text_for_contextual=self.corpus_og,
                                                  text_for_bow=self.corpus_pre)

        self.vocab = self.preparer.vocab

        return
    
    def train(self, num_topics: int = 5, num_epochs: int = 50) -> None:
        '''
        Main training loop of the class, will use the ZeroShotTM, initialize it
        and use the function .fit of the class.
        ------------
        Parameters:

        num_topics: int
            Number of topics for topic modelling
        
        num_epochs: int
            Number of epochs for training
        '''

        self.model = ZeroShotTM(bow_size=len(self.preparer.vocab),
                           contextual_size = self.training_dataset.X_contextual.shape[1],
                           n_components=num_topics,
                           num_epochs=num_epochs)
        
        #Main execution loop
        self.model.fit(self.training_dataset)

        #Storing parameters
        self.thetas = self.model.get_doc_topic_distribution(self.training_dataset)

        self.betas = self.model.get_topic_word_distribution()

        self.topics = self.model.get_topic_lists(10)
        
        return
    
    def create_folder_structure(self) -> None:
        '''
        Creates a folder structure like:
       
        - output_path:
            -- corpus_folder:
                ---- preproc_corpus.txt
                ---- original_corpus.txt
            -- zeroshot_output:
                ---- thetas.npy
                ---- betas.npy
                ---- vocab.txt
                ---- topic_lengths.npy
                ---- topics.json
        '''

        #Create the subdirectories
        dir_corpus = os.path.join(self.output_path, self.corpus_path)
        dir_ZS_out = os.path.join(self.output_path, self.ZS_output_path)

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
            self._logger.info(f'folder {self.output_path} created sucessfully!')

            os.mkdir(dir_corpus)
            self._logger.info(f'folder {dir_corpus} created sucessfully!')

            os.mkdir(dir_ZS_out)
            self._logger.info(f'folder {dir_ZS_out} created sucessfully!')

        else:
            if not os.path.exists(dir_corpus):
                os.mkdir(dir_corpus)
                self._logger.info(f'folder {dir_corpus} created sucessfully!')
            
            if not os.path.exists(dir_ZS_out):
                os.mkdir(dir_ZS_out)
                self._logger.info(f'folder {dir_ZS_out} created sucessfully!')
        return
    
    def save_results(self) -> None:
        '''
        Creates the folder structure and stores the different results in it namely:

        Both corpuses, the raw text and the lemmas stored as a .txt file

        The thetas and betas stored as .npy documents
        '''
        #Ensure the directories are created correctly
        self.create_folder_structure()

        #Obtain all the necessary results for the ldavis step

        results = self.model.get_ldavis_data_format(self.vocab,
                                                    self.training_dataset,
                                                    n_samples=10)

        #Check the existence of original_corpus.txt
        if os.path.exists(os.path.join(self.output_path,self.corpus_path,'original_corpus.txt')):
            self._logger.warning('Warning overwriting original corpus')

        with open('original_corpus.txt', 'w') as f:
            for element in self.corpus_og:
                f.write(f'{element}\n')

        if os.path.exists(os.path.join(self.output_path,self.corpus_path,'preprocessed_corpus.txt')):
            self._logger.warning('Warning overwriting preprocessed corpus')

        with open('preprocessed_corpus.txt', 'w') as f:
            for element in self.corpus_pre:
                f.write(f'{element}\n')

        #Now the outputs for pyLDAvis
        ZS_path = os.path.join(self.output_path, self.ZS_output_path)
        os.makedirs(ZS_path, exist_ok=True)

        # THETAS
        theta_path = os.path.join(ZS_path, 'thetas.npy')

        thetas = results['doc_topic_dists']
        np.save(theta_path, thetas)
        self._logger.info(f'Thetas saved in {theta_path}')

        # BETAS
        beta_path = os.path.join(ZS_path, 'betas.npy')

        betas = results['topic_term_dists']
        np.save(beta_path, betas)
        self._logger.info(f'Betas saved in {beta_path}')

        # TOPICS
        topic_path = os.path.join(ZS_path, 'topics.json')
        with open(topic_path, "w", encoding="utf-8") as f:
            json.dump(self.topics, f, indent=4, ensure_ascii=False)
        self._logger.info(f'Topics saved in {topic_path}')

        with open(topic_path, "r", encoding="utf-8") as f:
            topics = json.load(f)
        output_path = os.path.join(ZS_path, 'topics.txt')
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, words in enumerate(topics):
                word_str = " ".join(words)
                f.write(f"{idx}\t{word_str}\n")

        # DOC LENGTHS    
        doc_l_path = os.path.join(ZS_path, 'doc_len.npy')

        doc_len = results['doc_lengths']
        np.save(doc_l_path, doc_len)
        self._logger.info(f'Doc length saved in {doc_l_path}')

        # TERM DIST
        term_f_path = os.path.join(ZS_path, 'term_freq.npy')
        frequencies = results['term_frequency']
        np.save(term_f_path, frequencies)
        self._logger.info(f'Term distribution saved in {term_f_path}')

        # VOCAB
        vocab_path = os.path.join(ZS_path, 'vocab.txt')
        with open(vocab_path, 'w') as f:
            for word in self.vocab:
                f.write(f'{word}\n')
        
        return
    