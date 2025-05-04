import pandas as pd
import numpy as np
from tabulate import tabulate
from scipy import sparse
import os
import tqdm
import sys
import time
import textwrap
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.IRQ.indexer import *
from src.IRQ.query_eng import *
from src.mind.query_generator import QueryGenerator


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

        #self.query_engine = QueryEngine(       )
    
    def _resolve_model_path(self):
        # Just abstract it once
        dataset_name = 'en_2025-02-25_segmented_dataset.parquet.gzip'
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
            file_path = '/export/usuarios_ml4ds/ammesa/Data/3_joined_data/en_2025-02-25_segmented_dataset.parquet.gzip/polylingual_df'
            dataset = pd.read_parquet(file_path)
            self.dataset = dataset
        except FileNotFoundError:
            self.dataset=[]

        return

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_doc(self, doc):
        max_w = 80
        delay = 0.01
        wrapped_lines = textwrap.wrap(doc['raw_text'], width=max_w)

        for line in wrapped_lines:
            for char in line:
                print(char, end='', flush=True)
                time.sleep(delay)
            print()  # Newline after each wrapped line
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
            self.clear_screen()
            print(tabulate(menu, headers='keys', tablefmt='github', showindex=False))
            print("\n======================================================\n")
            subset = self.thetas_en[:,topic_number]
            top_n_idx_unsorted = np.argpartition(-subset, n_samp)[:n_samp]  
            top_n_idx_sorted = top_n_idx_unsorted[np.argsort(-subset[top_n_idx_unsorted])]  

            documents = self.dataset[self.dataset['doc_id'].isin(top_n_idx_sorted)]
            self.print_doc(documents.iloc[idx])
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

    def retrieval(self, topic_number:int):
        '''
        Initiates the IRQ process and retrieves contradictions for the main user
        '''
        #Obtain docs in which the selected topic is the most representative.
        subset = np.where(self.thetas_en[:, topic_number] == self.thetas_en.max(axis=1))[0]
        documents = self.dataset[self.dataset['doc_id'].isin(subset)]
        
        #save that dataframe to generate questions

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
                self.clear_screen()
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
                self.clear_screen()
                print(tabulate(topic_df, headers='keys', tablefmt='github', showindex=True))

                choice = input(f"\nSelect topic # (0-{len(self.topics)-1}) or [r] to return: ").strip()

                if choice.lower() == 'r':
                    cond = False
                    
                if choice.isdigit():
                    topic = int(choice)
                    if topic in range(len(self.topics)):
                        self.retrieval(topic_number=topic)
                else:
                    print('Invalid option chosen\n')

        elif option == '3':
            print('Entering debugging mode!\n')
        else:
            print('Exiting program!\n')
            self.clear_screen()
            result = False
    
        return result

    def main_loop(self):
        print('Initializing contradiction engine...')

        cond = True
        while cond:
            pd.DataFrame.from_dict(self.menu)

            self.clear_screen()
            print(tabulate(self.menu, headers='keys', tablefmt='github', showindex=False))
            try:
                choice = input('\nPlease select one from the following\n').strip()       
            except:
                raise(Exception)
            
            while choice not in self.menu['option']:
                try:
                    choice = input('Invalid option choose between [1,2,3,x]\n').strip()       
                except:
                    raise(Exception)
                        
            cond = self.select(choice)

        return

if __name__ == '__main__':
    cli = CLI()
    cli.main_loop()