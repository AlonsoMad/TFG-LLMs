import logging
import pathlib
import os
import faiss
from tqdm import tqdm
import ast
import time
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import sparse, stats
import dotenv
import scikit_posthocs as sp
from sentence_transformers import SentenceTransformer, util
from kneed import KneeLocator
from prompter.prompter import Prompter
from tabulate import tabulate
from scipy.ndimage import uniform_filter1d
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from .indexer import *
from mind.query_generator import QueryGenerator
from mind.question_generator import QuestionGenerator

dotenv.load_dotenv()

class QueryEngine(NLPoperator):
    '''
    Uses LLMs to handle all sorts of queries, in the user-machine interface
    and in the DB-Retriever interface.
    '''
    def __init__(self, file_path, model_path, model_name, config, q_path, topic_model, translation):
        super().__init__(file_path, model_path, model_name)

        # Config dictionary: to be defined
        default_config={}
        self.config = default_config if config is None else {**default_config, **config}
        self.q_path = q_path
        self.topic_model = topic_model
        # Translation specifies whether there's a mirrored copy of docs
        self.translation = translation

        self.indexer=Indexer(file_path=self.file_path,
                             model_path=self.model_path,
                             model_name=self.model_name,
                             config=self.config)
        
        self.retriever=Retriever(file_path=self.file_path,
                             model_path=self.model_path,
                             model_name=self.model_name,
                             question_path=q_path,
                             idx_config=self.config)
        
        self.query_engine=QueryGenerator()
        self.question_engine=QuestionGenerator()
        self._init_prompter()

        return
    
    def read_questions_file(self, n_sample: int, n_topic:int) -> None:
        '''
        Reads and saves as an atribute of the class the dataframe of questions
        provided by the user.
        '''
        print(f'Reading questions located at: {self.q_path} / topic_{n_topic}')
        if not os.path.exists(self.q_path):
            print(f'{self.q_path} not found! Exiting')
            return 
        
        if (n_sample <= 0) or n_sample is None:
            print(f'Invalid value for a sample, ensure it is bigger than 0!')
            return 
        
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
    
    def _init_prompter(self):
        '''
        Generates and intializes the prompter for answering
        '''
        llm_model = "gpt-4o-mini"
        ollama_host = "http://kumo02.tsc.uc3m.es:11434"

        prompter = Prompter(
            model_type=llm_model) 
            #,ollama_host=ollama_host)
        
        self.prompter = prompter
        return
    
    def index(self, bilingual, tm) -> None:
        '''
        Produces an index for the database
        '''
        self.indexer.index(bilingual=bilingual, topic_model=tm)
        return
    
    def generate_subqueries(self,row):
        return self.query_engine.generate_query(row['question'], row['raw_text'])
    

    def generate_questions(self,row):
        return self.question_engine.generate_question(row['raw_text'], row['raw_text'])
    
    def process_row(self, row, func):
        func_name = func.__name__  

        if 'subqueries' in func_name:
            col = 'subqueries'
        elif 'questions' in func_name:
            col = 'question'
        else:
            col = 'output'  # default fallback

        row[col] = func(row)
        return row
    
    def parallel_apply(self,df,func,max_workers=20):
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_row, row, func) for _, row in df.iterrows()]
            for future in as_completed(futures):
                results.append(future.result())
        return pd.DataFrame(results)
    

    def check_questions(self):
        '''
        Checks how many questions have been already answered, uses this
        to format a new dataframe of unanswered questions
        '''
        return
    
    def parse_questions(self, questions):
        return

    def generate_questions_queries(self, docs: pd.DataFrame):
        '''
        With a pd.Dataframe as input, uses LLMs to generate questions
        for the raw text that conform the documents. It then generates the necessary queries.
        '''
        # First generate the questions and assign relevant docs
        # Need for standard id
        if 'doc_id' in docs.columns:
            docs = docs.copy()  # avoid SettingWithCopyWarning
            docs['relevant_docs'] = docs['doc_id']

            enriched_docs = self.parallel_apply(docs, self.generate_questions, max_workers=20)
            enriched_docs = self.parallel_apply(enriched_docs, self.generate_subqueries, max_workers=20)

            enriched_docs['question'] = enriched_docs['question'].apply(lambda x: x[0].split('\n'))
            docs = enriched_docs

        return docs
    
    def save_questions(self, df:pd.DataFrame, n_topic:int):
        '''
        Provided a n_topic that showcases the topic corresponding to the selected docs
        it stores the dataframe as a xlsx in the q_path of the object
        '''
        saving_path = os.path.join(self.q_path,f'topic_{n_topic}',f'questions_len_{len(df)}')
        os.makedirs(saving_path, exist_ok=True)
        df.to_excel(os.path.join(saving_path, 'questions.xlsx'))
        return
    
    def save_results(self, df:pd.DataFrame, option:str):
        '''
        Saves the results as a readable file, if there is yet a previous file
        it appends the new results at the end of the same.
        '''

        path = os.getenv("OUTPUT_PATH", "/Data/mind_folder")
        date = datetime.today().strftime('%Y-%m-%d-%H-%M')
        if option=='answers':
            print('Saving the answers!')

            path_suffix = 'answered'
            path = os.path.join(path, path_suffix)
            os.makedirs(path, exist_ok=True)
            full_path = os.path.join(path, f'{date}_answered_files.parquet')
            if os.path.exists(full_path):
                df_aux = pd.read_parquet(full_path)
                df = pd.concat([df, df_aux])
            df.to_parquet(full_path)

        elif option=='final_results':
            print('Saving the results!')

            path_suffix = 'final_results'
            path = os.path.join(path, path_suffix)
            os.makedirs(path, exist_ok=True)
            full_path = os.path.join(path, f'{date}_results.parquet')
            if os.path.exists(full_path):
                df_aux = pd.read_parquet(full_path)
                df_new = df[~df['question_id'].isin(df_aux['question_id'])]
                df = pd.concat([df_aux, df_new])
            df.to_parquet(full_path)

        return
    
    def extend_to_full_sentence(self, text: str, num_words: int) -> str:
        text_in_words = text.split()
        truncated_text = " ".join(text_in_words[:num_words])
        remaining_text = " ".join(text_in_words[num_words:])
        period_index = remaining_text.find(".")
        if period_index != -1:
            extended_text = f"{truncated_text} {remaining_text[:period_index + 1]}"
        else:
            extended_text = truncated_text
        return re.sub(r'\s([?.!,"])', r'\1', extended_text)

    
    def give_answer(self, df:pd.DataFrame):
        dotenv.load_dotenv()

        path_to_templates = os.getenv("MIND_TEMPLATES_PATH", "/src/mind/templates")

        _3_INSTRUCTIONS_PATH = os.path.join(path_to_templates, 'question_answering.txt') 
        _4_INSTRUCTIONS_PATH = os.path.join(path_to_templates, 'discrepancy_detection.txt') 
        RELEVANCE_PROMPT = os.path.join(path_to_templates, 'test_relevance.txt')  
        #Takes automatically the raw texts that were not used to do the initial retrieving
        raw = self.retriever.raw_o_lang
        aux_df = self.retriever.raw_lang
        df['question'] = df['question'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

        df_grouped = (
            df.groupby('question')
            .agg({
                'doc_id': lambda x: list(x)[0],
                #'full_doc': lambda x: list(x)[0],
                #'passage': lambda x: list(x)[0],
                'raw_text': lambda x: list(x)[0],
                'subqueries': lambda x: list(x)[0],
                'all_results': lambda x: list(x),
                'relevant_docs': lambda x: list(x)[0],
                'all_results_content': lambda x: list(x),
                'time': lambda x: list(x)[0]
            })
            .reset_index()
        )
        df_grouped['question_id'] = df_grouped['doc_id']

        for _, row in tqdm(df_grouped.iterrows(), total=len(df_grouped)):
            try:
                results = []

                n_top = self.retriever.config['top_k']
                top_docs = row['all_results'][:n_top]
                is_lang_en = str(self.retriever.raw[self.retriever.raw['doc_id']==row.doc_id].id_preproc.values[0]).startswith(("EN", "T_EN"))

                if is_lang_en:
                    raw = self.retriever.raw_o_lang
                    aux_df = self.retriever.raw_lang
                else: 
                    raw = self.retriever.raw_lang
                    aux_df = self.retriever.raw_o_lang
                '''
                results_4_unweighted = row[self.config['match']]
                flattened_list = [{'doc_id': entry['doc_id'], 'score': entry['score']} for subarray in results_4_unweighted for entry in subarray]
                top_docs = [el["doc_id"] for el in flattened_list][:self.config['r
                x
                ']]
                '''

                for top_doc in top_docs:
                
                    # ---------------------------------------------------------#
                    # 3. ANSWER GENERATION
                    #----------------------------------------------------------#
                    with open(_3_INSTRUCTIONS_PATH, 'r') as file: template = file.read()
                    ######################################
                    # GENERATE ANSWER IN SOURCE LANGUAGE 
                    ######################################
                    passage_s = row.raw_text
                    full_doc_s = row.raw_text
                    
                    formatted_template = template.format(question=row.question, passage=passage_s,full_document=(self.extend_to_full_sentence(full_doc_s, 100)+ " [...]"))
                    
                    answer_s, _ = self.prompter.prompt(question=formatted_template)
                    print("Answer S:", answer_s)
                    

                    id_prep = aux_df[aux_df['doc_id'] == top_doc]['id_preproc'].values[0]

                    if self.translation:
                        if not id_prep.startswith('T_'):
                            new_id_prep = f'T_{id_prep}'
                        elif id_prep.startswith('T_EN'):
                            new_id_prep = id_prep.replace('T_EN', 'EN', 1)
                        elif id_prep.startswith('T_ES'):
                            new_id_prep = id_prep.replace('T_ES', 'ES', 1)
                    else: 
                        first_raw_id = raw.iloc[0]['id_preproc']
                        match = re.match(r'([A-Z]+)_(\d+)_\d+', first_raw_id)
                        if match:
                            _, docnum_raw = match.groups()
                            docnum_raw = int(docnum_raw)
                        else:
                            raise ValueError(f"Invalid format in raw id_preproc: {first_raw_id}")

                        match = re.match(r'([A-Z]+)_(\d+)_(\d+)', id_prep)

                        if match:
                            language_prep, docnum_prep, paragraph_prep = match.groups()
                        else:
                            raise ValueError(f"Invalid format in id_prep: {id_prep}")


                        language_switched = 'EN' if language_prep == 'ES' else 'ES'

                        # 4. Adjust DOCNUM
                        if language_switched == 'EN':
                            new_docnum = (docnum_prep[1:].lstrip('0') or '0') if len(docnum_prep) > 1 else '0'
                        else:
                            new_docnum = int(docnum_prep) + docnum_raw

                        if int(new_docnum) < 0:
                            raise ValueError("Resulting DOCNUM is negative after subtraction.")

                        new_id_prep = f"{language_switched}_{new_docnum}_{paragraph_prep}"
                        print(f"Mapped id_prep: {new_id_prep}")


                    ######################################
                    # GENERATE ANSWER IN TARGET LANGUAGE #
                    ######################################

                    # if row.doc_id in top_docs:
                        # import pdb; pdb.set_trace()
                    try:
                        passage_t = raw[raw.id_preproc == new_id_prep].raw_text.iloc[0]
                        full_doc_t = raw[raw.id_preproc == new_id_prep].raw_text.iloc[0]
                    except IndexError:
                        passage_t = 'There is no information in the database to answer'
                        full_doc_t = 'There is no information in the database to answer'

                    

                    ##############################################
                    # CHECK RELEVANCE OF PASSAGE TO THE QUESTION #
                    ##############################################x
                    with open(RELEVANCE_PROMPT, 'r') as file: template = file.read()
                    formatted_template = template.format(passage=passage_t, question=row.question)
                    
                    response, _ = self.prompter.prompt(question=formatted_template)
                    relevance = 1 if "yes" in response.lower() else 0
                    
                    if relevance == 0:
                        answer_t = "I cannot answer the question given the context."
                    else:
                        with open(_3_INSTRUCTIONS_PATH, 'r') as file: template = file.read()
                    
                        formatted_template = template.format(question=row.question, passage=passage_t,full_document=(self.extend_to_full_sentence(full_doc_t, 100)+ " [...]"))
                        answer_t, _ = self.prompter.prompt(question=formatted_template)
                    
                    print("Answer T:", answer_t)
                    
                    if "cannot answer the question given the context" not in answer_t:
                        #-----------------------------------------------------#
                        # 4. DISCREPANCY DETECTION
                        # ------------------------------------------------------#
                        with open(_4_INSTRUCTIONS_PATH, 'r') as file: template = file.read()
                        
                        question = template.format(question=row.question, answer_1=answer_s, answer_2=answer_t)
                        
                        discrepancy, _ = self.prompter.prompt(question=question)
                        
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

                        # if label == 'CULTURAL_DISCREPANCY':
                        #     import pdb; pdb.set_trace()

                        
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
                # Cambiar esto, no debe guardar en checkpoint
                self.save_results(checkpoint_df, 'final_results')
                    
            except Exception as e:
                print(f"Error with question {row.question_id}: {e}")
                continue



        return

    def disc_det(self, df: pd.DataFrame):
        '''
        Uses an LLM to check whether two answers are contradictory or not
        also checks for absence of an answer
        '''
        # This function is not used in the current implementation
                #-----------------------------------------------------#
        # 4. DISCREPANCY DETECTION
        # ------------------------------------------------------#
        _4_INSTRUCTIONS_PATH = "/export/usuarios_ml4ds/ammesa/TFG-LLMs/src/mind/templates/discrepancy_detection.txt"
        with open(_4_INSTRUCTIONS_PATH, 'r') as file: template = file.read()
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df)):

            answer_s = row.answer_s
            answer_t = row.answer_t
            passage_s = row.passage_s
            passage_t = row.passage_t

            question = template.format(question=row.question, answer_1=answer_s, answer_2=answer_t)
            
            discrepancy, _ = self.prompter.prompt(question=question)
            if "cannot answer the question given the context" not in answer_t:
            
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
                "question": row.question,
                "passage_s": passage_s,
                "answer_s": answer_s,
                "passage_t": passage_t,
                "answer_t": answer_t,
                "discrepancy": label,
                "reason": reason,
                "model" : self.prompter.model_type
            })

        # Save checkpoint
        checkpoint_df = pd.DataFrame(results)
        self.save_results(checkpoint_df, 'final_results')

        return

    def check_discrepancies(self):
        '''
        Uses an LLM to check whether two answers are contradictory or not
        also checks for absence of an answer
        '''
        return

    def answer_loop(self,input_df:pd.DataFrame, topics:int, parallel:bool):
        '''
        Integrates previous functionalities in an answer loop
        '''
        #Recuerda que self raw de retriever debe ser el df en otro idioma

        self.retriever.retrieval_loop(bilingual=True, n_tpcs=topics,
                                       topic_model=self.topic_model,weight=True, question_df=input_df,
                                       parallel=parallel)
        
        answer_df = self.retriever.output_df
        self.save_results(answer_df, 'answers')

        self.give_answer(answer_df)

        return
    

    