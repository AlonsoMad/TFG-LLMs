'''
Python class for the WikiRetriever object. Given a System output path and a number of documents creates a .parquet dataset in that path with the specified length.
Documents limited to english and spanish. Originally the seed page is George Washington (no desambiguation needed), it can be changed to anything
'''
from tqdm import tqdm
import wikipediaapi as wiki
import pandas as pd
import numpy as np
import os
from datetime import date

class WikiRetriever():
  '''
  Class to carry out the dataset creation for the wikipedia retrieval
  '''

  def __init__(self,
               file_path: str,
               seed_lan: str = "en",
               seed_query: str = "George Washington",
               project_name: str = 'AlonsoMadBot/0.0 (100454449@alumnos.uc3m.es) wikipedia-api/0.8.1' ,
               #When implemented change to 1000
               ndocs: int = 0,
               #Max size of the stack of to-read docs
               max_size: int = 1000,

               doc_en_cnt: int = 0,
               doc_es_cnt: int = 0,
               next_doc_stack: list = None,
               en_docs_df: pd.DataFrame = None,
               es_docs_df: pd.DataFrame = None,
               agent: wiki.Wikipedia = None
               ):

    self.file_path = file_path
    self.seed_lan = seed_lan
    self.seed_query = seed_query
    self.project_name = project_name
    self.ndocs = ndocs

    self.max_size = max_size
    self.doc_en_cnt = doc_en_cnt
    self.doc_es_cnt = doc_es_cnt

    #Set automatically the lists and dataframes
    self.next_doc_stack = next_doc_stack if next_doc_stack is not None else []

    self.en_docs_df = en_docs_df if en_docs_df is not None else pd.DataFrame(columns=["title",
                                                                                      "summary",
                                                                                      "text",
                                                                                      "lang",
                                                                                      "url",
                                                                                      "equivalence"])

    self.es_docs_df = es_docs_df if es_docs_df is not None else pd.DataFrame(columns=["title",
                                                                                      "summary",
                                                                                      "text",
                                                                                      "lang",
                                                                                      "url",
                                                                                      "equivalence"])
    
    self.en_doc_list = []
    self.es_doc_list = []
    

    self.agent = agent if agent is not None else wiki.Wikipedia(user_agent=self.project_name, language=self.seed_lan)

    self.final_file_name = ''

    return

  def update_stack(self, titles: dict) -> None:
    '''
    Inserts in the next document list as many titles as possible without
    overflow

    Parameters
    -----------
    titles: dict
      The dictionary of titles obtained from the API
    '''

    titles = list(titles)

    #Avoid taking more titles than needed
    window = self.max_size - len(self.next_doc_stack)

    #Take as maximum the available space in the stack
    titles = titles[:window]

    # concatenate the results
    self.next_doc_stack += titles

    return
  
  def update_df_notaligned(self, title:str) -> None:
    '''
    In case of needing to update the dataframes unbalanced
    this function will append new entries alternating the 
    two of them English and Spanish. Apart from that its 
    function its the same than its sister.
    '''
    try:
      query = self.agent.page(title)
      #checks if the page is already in the collection
      repeated_en = self.en_docs_df["title"].eq(query.title).any()
      repeated_es = self.es_docs_df["title"].eq(query.title).any()

      condition = (repeated_es or repeated_en)

      if query.exists() and "es" in query.langlinks.keys() and not condition:
        query_esp = query.langlinks["es"]
        #check which lang is more populated
        if self.doc_en_cnt >= self.doc_es_cnt:
          #if it is english go with spanish
          self.es_doc_list.append([query_esp.title,
                                    query_esp.summary,
                                    query_esp.text,
                                    "es",
                                    query.fullurl,
                                    0])
          
          self.doc_es_cnt += 1
          if len(self.es_doc_list) >= 1000:
            temp_df = pd.DataFrame(
                  self.es_doc_list,
                  columns=["title", "summary", "text", "lang", "url", "label"]
              )
            self.es_docs_df = pd.concat([self.es_docs_df, temp_df], ignore_index=True)

            # Clear the list after appending
            self.es_doc_list.clear()

        else:
          #else choose english
          self.en_doc_list.append([query.title,
                                    query.summary,
                                    query.text,
                                    "en",
                                    query.fullurl,
                                    0])
          
          self.doc_en_cnt += 1
          if len(self.en_doc_list) >= 1000:
            temp_df = pd.DataFrame(
                  self.en_doc_list,
                  columns=["title", "summary", "text", "lang", "url", "label"]
              )
            self.en_docs_df = pd.concat([self.en_docs_df, temp_df], ignore_index=True)

            # Clear the list after appending
            self.en_doc_list.clear()

      #whatever it is choose to populate the stack if needed
        if self.max_size - len(self.next_doc_stack) >= 800:
            next_titles = query.links.keys()
            self.update_stack(next_titles)
    except:
        print('Network exception ocurred')

    return
  
  def update_dataframes(self, title:str) -> None:
    '''
    Given the title of an entry, checks existence, bilinguality, repetition
    and appends it.

    Parameters
    -------------
    title: str
      The title of the page to append
    '''
    #Get new query
    query = self.agent.page(title)

    #checks if the page is already in the collection
    repeated = self.en_docs_df["title"].eq(query.title).any()

    if query.exists() and not repeated:
      #check that both languages are available
      if "es" in query.langlinks.keys():
        #Update english dict
        #else choose english
        self.en_doc_list.append([query.title,
                                  query.summary,
                                  query.text,
                                  "en",
                                  query.fullurl,
                                  0])
        
        self.doc_en_cnt += 1
        if len(self.es_doc_list) >= 1000:
          temp_df = pd.DataFrame(
                self.en_doc_list,
                columns=["title", "summary", "text", "lang", "url", "label"]
            )
          self.en_docs_df = pd.concat([self.en_docs_df, temp_df], ignore_index=True)

          # Clear the list after appending
          self.en_doc_list.clear()


        #Update spanish
        query_esp = query.langlinks["es"]
        self.es_doc_list.append([query_esp.title,
                                  query_esp.summary,
                                  query_esp.text,
                                  "es",
                                  query.fullurl,
                                  0])
        
        self.doc_es_cnt += 1
        if len(self.en_doc_list) >= 1000:
          temp_df = pd.DataFrame(
                self.es_doc_list,
                columns=["title", "summary", "text", "lang", "url", "label"]
            )
          self.es_docs_df = pd.concat([self.es_docs_df, temp_df], ignore_index=True)

          # Clear the list after appending
          self.es_doc_list.clear()
        
        #If statement to update the stack only when its at 20% capacity
        if self.max_size - len(self.next_doc_stack) >= 800:
          next_titles = query.links.keys()
          self.update_stack(next_titles)
    return


  def append_df(self, lang:str):
    if lang == 'es':
      temp_df = pd.DataFrame(
        self.es_doc_list,
        columns=["title", "summary", "text", "lang", "url", "label"]
      )
      self.es_docs_df = pd.concat([self.es_docs_df, temp_df], ignore_index=True)

      # Clear the list after appending
      self.es_doc_list.clear()
    else:
      temp_df = pd.DataFrame(
        self.en_doc_list,
        columns=["title", "summary", "text", "lang", "url", "label"]
      )
      self.en_docs_df = pd.concat([self.en_docs_df, temp_df], ignore_index=True)

      # Clear the list after appending
      self.en_doc_list.clear()

    return

  def retrieval(self, alignment: float = 1) -> None:
    '''
    Main loop of the function, will recursively iterate until reaching
    maxdepth called Ndocs.

    Parameters
    ------------
    Ndocs: int
      Number of documents per language, the final dataset is 2x this number

    seed_query: str
      The first wikipedia page that will be inserted in the dataset

    aligment: float
      Percentage of shared documents between the two languages
    '''
    if alignment > 1 or alignment < 0:
      raise Exception('Alignment must be bound [0,1]')
    
    else:
      import pdb; pdb.set_trace()
      #get the number of different docs
      ndocs_notalign = int((1-alignment)*self.ndocs)
      #get each doc in one or other language
      while (self.doc_en_cnt + self.doc_es_cnt) < ndocs_notalign:
        if (self.doc_en_cnt + self.doc_es_cnt) == 0:
          self.update_df_notaligned(self.seed_query)
        else:
          new_title = self.next_doc_stack.pop(0)
          self.update_df_notaligned(new_title)
        
        if (self.doc_en_cnt + self.doc_es_cnt)%10000 == 0:
          self.df_to_parquet()

      #until completion
      with tqdm(total=int(self.ndocs/2)) as pbar:

        while self.doc_en_cnt < int(self.ndocs/2):
          #handle first case
          if self.doc_en_cnt == 0:
            self.update_dataframes(self.seed_query)

          else:
            new_title = self.next_doc_stack.pop(0)
            self.update_dataframes(new_title)

          pbar.n = self.doc_en_cnt
          pbar.refresh()

          if (self.doc_en_cnt + self.doc_es_cnt)%10000 == 0:
            self.df_to_parquet()
      
      self.append_df('en')
      self.append_df('es')

    return
  
  def restart(self, alignment:float, path:str) -> None:
    '''
    In case of a failed execution, opens the chosen dataset,
    picks up the last entry as seed and restarts the process
    '''
    aux_df = pd.read_parquet(path)

    self.en_docs_df = aux_df[aux_df['lang'] == 'en']
    self.es_docs_df = aux_df[aux_df['lang'] == 'es']

    self.doc_en_cnt = len(self.en_docs_df)
    self.doc_es_cnt = len(self.es_docs_df)

    query = self.agent.page(self.seed_query)
    next_titles = query.links.keys()
    self.update_stack(next_titles)
    self.retrieval(alignment)

    return
  

  def df_to_parquet(self, collab:bool = False) -> None:
    '''
    Merges the EN and ES dataframes and transforms them to .parquet
    then downloads it.

    Parameters
    -----------
    Collab:bool
      Saves the files if working in the cloud, if False directly downloads them

    '''
    df = pd.concat([self.en_docs_df, self.es_docs_df],
                           ignore_index=True)
    
    #create the unique identifier
    df["id"] = df.index

    #Preprocessing identifier
    df["id_preproc"] = df["lang"].str.upper() + "_" + df["id"].astype(str)

    date_name = str(date.today())

    self.final_file_name = f"dataset_{date_name}.parquet.gzip"

    save_path = os.path.join(self.file_path, self.final_file_name)

    if collab:
      if "drive" not in os.listdir("/content"):
        from google.colab import drive
        drive.mount('/content/drive')

      df.to_parquet(path=save_path, compression="gzip")
      print(f"Saving in Drive: {save_path}")

    else:
      df.to_parquet(path=save_path, compression="gzip")
      print(f"Saving in PC: {save_path}")

    return
