import pandas as pd
import numpy as np
from tabulate import tabulate
from scipy import sparse
import os
import tqdm
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.IRQ.indexer import *
from src.IRQ.query_eng import *
from src.mind.query_generator import QueryGenerator
from src.utils.utils import clear_screen, print_doc

class CLI:
    def __init__(self):
        self.topics = []
        self.model_path = self._resolve_model_path()
        self._load_topics()
        self._load_docs()
        self.menu = {
                    'option':['1','2','3','x'],
                    'description':['Inspect topics',
                                    'Select topic for retrieval',
                                    'Debug (not-implemented)',
                                    'Exit menu' ]
                    }

        self.q_engine = None
    
    def _resolve_model_path(self):
        # Just abstract it once
        dataset_name = "en_2025-03-03_segm_trans"
        return os.path.join('/export/usuarios_ml4ds/ammesa/mallet_folder', dataset_name, 'mallet_output')

    def _load_topics(self):
        try:
            with open(os.path.join(self.model_path, 'keys_EN.txt')) as f:
                self.topics = [line.strip() for line in f]
        except FileNotFoundError:
            self.topics = []

        try:
            thetas_path = os.path.join(self.model_path, f'thetas_EN.npz')
            self.thetas_en = sparse.load_npz(thetas_path).toarray()
            thetas_path = os.path.join(self.model_path,f'thetas_ES.npz')
            self.thetas_es = sparse.load_npz(thetas_path).toarray()
        except FileNotFoundError:
            self.thetas_en = None
            self.thetas_es = None


    def _load_docs(self):
        try:
            file_path = '/export/usuarios_ml4ds/ammesa/Data/3_joined_data/en_2025-03-03_segm_trans/polylingual_df'
            dataset = pd.read_parquet(file_path)
            self.dataset = dataset
        except FileNotFoundError:
            self.dataset=[]

        return

    def _init_q_en(self, q_path:str):
        '''
        Initializes the query generator and handler
        '''
        config={  
            "match": 'TB_ENN',
            "embedding_size": 384,
            "min_clusters": 8,
            "top_k_hits": 10,
            "batch_size": 32,
            "thr": '0.01',
            "top_k": 10,
            'storage_path': '/export/usuarios_ml4ds/ammesa/Data/4_indexed_data'
        }

        qg = QueryEngine(
            file_path='/export/usuarios_ml4ds/ammesa/Data/3_joined_data/en_2025-03-03_segm_trans/polylingual_df',
            model_path='/export/usuarios_ml4ds/ammesa/mallet_folder/en_2025-03-03_segm_trans',
            model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            config=config,
            q_path=q_path
        )
        self.q_engine = qg

        return


    def inspect_topics(self, topic_number:int):
        #Search for the most representative doc of that topic
        n_samp = 10
        cond = True
        menu = {
                'option':['n','p','r'],
                'description':['Next document',
                                'Previous document',
                                'Return' ]
                }

        pd.DataFrame.from_dict(menu)
        idx = 0
        while cond:
            clear_screen()
            print(tabulate(menu, headers='keys', tablefmt='github', showindex=False))
            print("\n======================================================\n")
            subset = self.thetas_en[:,topic_number]
            top_n_idx_unsorted = np.argpartition(-subset, n_samp)[:n_samp]  
            top_n_idx_sorted = top_n_idx_unsorted[np.argsort(-subset[top_n_idx_unsorted])]  

            documents = self.dataset[self.dataset['doc_id'].isin(top_n_idx_sorted)]
            print_doc(documents.iloc[idx])
            option = input('').strip()
            while option not in set(menu['option']):
                option = input('\nWrong option, choose [n,p,r]\n')

            if option == 'n':
                idx += 1
            elif option == 'p':
                idx -= 1
            else:
                cond = False

        return

    def retrieval(self, topic_number:int, n_sample:int):
        '''
        Initiates the IRQ process and retrieves contradictions for the main user
        
        - topic_number: int
            The number associated with the topic of the analyzed documents
        '''
        clear_screen()
        #Obtain docs in which the selected topic is the most representative.
        subset = np.where(self.thetas_en[:, topic_number] == self.thetas_en.max(axis=1))[0]
        documents = self.dataset[self.dataset['doc_id'].isin(subset)]
        
        documents = documents.sample(n_sample)

        #save that dataframe to generate questions
        path = f'/export/usuarios_ml4ds/ammesa/mind_folder/question_bank'
        os.makedirs(path, exist_ok=True)

        if self.q_engine == None:
            self._init_q_en(q_path=path)
        
        path = os.path.join(path,f'topic_{topic_number}', f'questions_len_{n_sample}')
        self.q_engine.retriever.update_q_path(path=path)

        df_aux = self.q_engine.generate_questions_queries(documents)
        self.q_engine.save_questions(df_aux, topic_number)

        self.q_engine.answer_loop(df_aux, topic_number)

        return

    def select(self,option:str):
        '''
        Handles the option selected by the user
        '''
        result = True
        if option == '1':
            print('Inspecting topics\n')
            sample_topics = [' '.join(topic.split()[:3]) for topic in self.topics]
            topic_df = pd.DataFrame(
                sample_topics,
                index = range(len(self.topics)),
                columns=['Key words']
            )

            cond = True
            while cond:
                clear_screen()
                print(tabulate(topic_df, headers='keys', tablefmt='github', showindex=True))

                choice = input(f"\nSelect topic # (0-{len(self.topics)-1}) or [r] to return: ").strip()

                if choice.lower() == 'r':
                    cond = False
                    
                if choice.isdigit():
                    topic = int(choice)
                    if topic in range(len(self.topics)):
                        self.inspect_topics(topic_number=topic)
                        #cond = False
                else:
                    print('Invalid option chosen\n')

        elif option == '2':
            print('Selecting topics for retrieval\n')
            sample_topics = [' '.join(topic.split()[:3]) for topic in self.topics]
            topic_df = pd.DataFrame(
                sample_topics,
                index = range(len(self.topics)),
                columns=['Key words']
            )

            cond = True
            while cond:
                clear_screen()
                print(tabulate(topic_df, headers='keys', tablefmt='github', showindex=True))

                choice = input(f"\nSelect topic # (0-{len(self.topics)-1}) or [r] to return: ").strip()

                if choice.lower() == 'r':
                    cond = False
                    
                if choice.isdigit():
                    topic = int(choice)
                    if topic in range(len(self.topics)):

                        n_sample = input(f'\nSelect the number of samples to analyze: ').strip()
                        while not n_sample.isdigit() and n_sample != 'r':
                            n_sample = input(f'\nSelect an int number of samples or return [r]: ').strip()
                        
                        if n_sample == 'r':
                            
                            cond = False
                        elif int(n_sample) > len(np.where(self.thetas_en[:, topic] == self.thetas_en.max(axis=1))[0]):
                            n_sample=len(np.where(self.thetas_en[:, topic] == self.thetas_en.max(axis=1))[0])
                            self.retrieval(topic_number=topic, n_sample=n_sample)

                        else:
                            n_sample=int(n_sample)
                            self.retrieval(topic_number=topic, n_sample=n_sample)

                else:
                    print('Invalid option chosen\n')

        elif option == '3':
            print('Entering debugging mode!\n')
        else:
            print('Exiting program!\n')
            clear_screen()
            result = False
    
        return result

    def main_loop(self):
        cond = True
        while cond:
            pd.DataFrame.from_dict(self.menu)

            clear_screen()
            print(tabulate(self.menu, headers='keys', tablefmt='github', showindex=False))
            
            choice = input('\nPlease select one from the following\n').strip()       

            while choice not in self.menu['option']:
                choice = input('Invalid option choose between [1,2,3,x]\n').strip()       
                
                        
            cond = self.select(choice)

        return

if __name__ == '__main__':
    cli = CLI()
    try:
        cli.main_loop()
    except KeyboardInterrupt:
        print('\nExiting')