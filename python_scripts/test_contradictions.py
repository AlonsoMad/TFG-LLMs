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
from src.prompter.prompter import Prompter
from tabulate import tabulate
from scipy.ndimage import uniform_filter1d
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.IRQ.indexer import *
from src.mind.query_generator import QueryGenerator
from src.mind.question_generator import QuestionGenerator


_3_INSTRUCTIONS_PATH = "/export/usuarios_ml4ds/ammesa/TFG-LLMs/src/mind/templates/question_answering.txt"
_4_INSTRUCTIONS_PATH = "/export/usuarios_ml4ds/ammesa/TFG-LLMs/src/mind/templates/discrepancy_detection.txt"
RELEVANCE_PROMPT = "/export/usuarios_ml4ds/ammesa/TFG-LLMs/src/mind/templates/test_relevance.txt"
#Takes automatically the raw texts that were not used to do the initial retrieving

llm_model = 'qwen3:32b'# "qwen:32b"
ollama_host = "http://kumo01.tsc.uc3m.es:11434"
prompter = Prompter(model_type=llm_model) 

df = pd.read_csv('/export/usuarios_ml4ds/ammesa/Data/question_bank/q_fev_questions/FEVER-DPLACE-Q_v2_discp.csv')
results = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
    
        answer_1=row.answer1
        answer_2=row.answer2
        if "cannot answer the question given the context" not in answer_2:
            #-----------------------------------------------------#
            # 4. DISCREPANCY DETECTION
            # ------------------------------------------------------#
            with open(_4_INSTRUCTIONS_PATH, 'r') as file: template = file.read()
            
            question = template.format(question=row.question, answer_1=answer_1, answer_2=answer_2)
            
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
            if answer_2 == "I cannot answer as the passage contains personal opinions.":
                reason = "I cannot answer as the passage contains personal opinions."
                label = "NOT_ENOUGH_INFO"
            else:
                reason = "I cannot answer given the context."
                label = "NOT_ENOUGH_INFO"
            
            
        results.append({
            "claim": row.claim,
            "question": row.question,
            'evidence':row.evidence,
            "answer_s": answer_1,
            "answer_t": answer_2,
            "label": row.label,
            'discp_qwen:32b':label,
            "reason": reason
        })
    except:
        a = []

import pdb; pdb.set_trace()
results_df = pd.DataFrame(results)
save_path = os.path.join('/export/usuarios_ml4ds/ammesa/Data/question_bank/q_fev_questions', "results.csv")  # or .parquet

# Save as CSV
results_df.to_csv(save_path, index=False)

