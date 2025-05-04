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
from .indexer import *
from src.mind.query_generator import QueryGenerator


class QueryEngine(NLPoperator):
    '''
    Uses LLMs to handle all sorts of queries, in the user-machine interface
    and in the DB-Retriever interface.
    '''
    def __init__(self, file_path, model_path, model_name, config, q_path):
        super().__init__(file_path, model_path, model_name)

        # Config dictionary: to be defined
        default_config={}
        self.config = default_config if config is None else {**default_config, **config}
        self.q_path = q_path

        self.indexer=Indexer(file_path=self.file_path,
                             model_path=self.model_path,
                             model_name=self.model_name,
                             config=self.config)
        
        self.retriever=Retriever(file_path=self.file_path,
                             model_path=self.model_path,
                             model_name=self.model_name,
                             question_path=q_path,
                             config=self.config)
        
        self.question_engine=QueryGenerator()

        return
    
    def read_questions_file(self, n_sample: int) -> None:
        '''
        Reads and saves as an atribute of the class the dataframe of questions
        provided by the user.
        '''
        print(f'Reading questions located at: {self.q_path}')
        if not os.path.exists(self.q_path):
            print(f'{self.q_path} not found! Exiting')
            return -1
        
        if (n_sample <= 0) or n_sample is None:
            print(f'Invalid value for a sample, ensure it is bigger than 0!')
            return -1
        
        q_excel = pd.read_excel(self.q_path)

        #Filter out processed
        file_name = os.path.basename(self.q_path)
        path_prev = os.path.join(self.storage_path, file_name)
        if os.path.exists(path_prev):
            print('Loading previous results')

            q_prev = pd.read_parquet(path_prev)
            q_all = q_excel[~q_excel['question_id'].isin(q_prev['question_id'])]
            print(f"Already processed {q_prev.shape[0]} queries. Remaining: {q_all.shape[0]}")

        df_q_all = q_all.sample(n_sample, random_state=42)
        self.question_df = df_q_all

        return
    
    def index(self, bilingual, tm) -> None:
        '''
        Produces an index for the database
        '''
        self.indexer.index(bilingual=bilingual, topic_model=tm)
        return
    
    def check_questions(self):
        '''
        Checks how many questions have been already answered, uses this
        to format a new dataframe of unanswered questions
        '''
        return
    
    def generate_questions(self, docs: pd.DataFrame):

        return
    
    def save_results(self):
        '''
        Saves the results as a readable file
        '''
        return
    
    def check_discrepancies(self):
        '''
        Uses an LLM to check whether two answers are contradictory or not
        also checks for absence of an answer
        '''
        return

    def answer_loop(self):
        '''
        Integrates previous functionalities in an answer loop
        '''

        
        return
    
    def initialize_interface(self) -> None:
        '''
        Starts communication with the user by a CLI,
        registers the different inputs of the user
        '''


        return
    