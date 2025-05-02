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




class QueryEngine(NLPoperator):
    '''
    Uses LLMs to handle all sorts of queries, in the user-machine interface
    and in the DB-Retriever interface.
    '''
    def __init__(self, file_path, model_path, model_name, thr, top_k):
        super().__init__(file_path, model_path, model_name, thr, top_k)
        return
    
    def read_questions_file(self):
        '''
        Reads and saves as an atribute of the class the dataframe of questions
        provided by the user
        '''
        return
    
    def check_questions(self):
        '''
        Checks how many questions have been already answered, uses this
        to format a new dataframe of unanswered questions
        '''
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
    