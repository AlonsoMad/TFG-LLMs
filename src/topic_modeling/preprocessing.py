import pandas as pd
import os
import logging
from datetime import date
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from datasets import Dataset

class Segmenter():
    '''
    Class to carry out the segmentation of the dataset
    generated with wiki-retriever

    Parameters
    -------------
    in_directory: str
        Directory where the input file can be found
            /export/users/DATA
    file_name: str
        Name of the file itself
            mydata.parquet.gzip
    '''
    def __init__(self,
                 in_directory: str,
                 file_name: str,
                 out_directory: str,
                 input_df: pd.DataFrame = None,
                 segmented_df: pd.DataFrame = None,
                 logger: logging.Logger = None
                 ):
        
        self.in_directory = in_directory
        self.file_name = file_name
        self.out_directory = out_directory

        self.en_df = None
        self.es_df = None

        self.input_df = input_df if input_df is not None else pd.DataFrame(columns=["title",
                                                                                    "summary",
                                                                                    "text",
                                                                                    "lang",
                                                                                    "url",
                                                                                    "id",
                                                                                    "equivalence",
                                                                                    "id_preproc"])
        
        self.segmented_df = segmented_df if segmented_df is not None else pd.DataFrame(columns=["title",
                                                                                    "summary",
                                                                                    "text",
                                                                                    "lang",
                                                                                    "url",
                                                                                    "id",
                                                                                    "equivalence",
                                                                                    "id_preproc"])
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


    def read_dataframe(self) -> None:
        '''
        Checks the in_directory path and
        saves the file as a PD dataframe (input_df)
        '''
        if not os.path.exists(self.in_directory):
          raise Exception('Path not found, check again')

        elif not os.path.isfile(os.path.join(self.in_directory, self.file_name)):
          raise Exception('File not found, check again')

        else:
          self.input_df = pd.read_parquet(os.path.join(self.in_directory, self.file_name))
          self._logger.info("File read sucessfully!")
        return
    
    def segment(self) -> None:
        '''
        Iterates over the input dataset and creates a second one in which each
        entry represents a paragraph of the original
        '''
        self._logger.info("Starting segmentation!")
        for i, row in tqdm(self.input_df.iterrows(), total=len(self.input_df), desc="Processing rows"):

          #Separates each text over the paragraph
          split_text = row['text'].split("\n")
          #Filters section names and blanks 
          filtered_text = filter(lambda x: x != '' and len(x) > 100 ,split_text)

          paragraphs = list(filtered_text)

          for _, p in enumerate(paragraphs):
            #Add the new data
            self.segmented_df.loc[len(self.segmented_df)] = [row['title'],
                                              row['summary'],
                                              p,
                                              row['lang'],
                                              row['url'],
                                              len(self.segmented_df),
                                              row["equivalence"],
                                              row['id_preproc']+"_"+str(_)]
        

        self.en_df = self.segmented_df[self.segmented_df['lang'] == 'en']
        self.es_df = self.segmented_df[self.segmented_df['lang'] == 'es']

        #TODO: implement machine translation functionality

        self.save_to_parquet(self.en_df, 'en')
        self.save_to_parquet(self.es_df, 'es')

        return
        
    def save_to_parquet(self, df: pd.DataFrame, lang: str, collab:bool = False) -> None:
    
      date_name = str(date.today())

      file_name = f"{lang}_{date_name}_segmented_dataset.parquet.gzip"

      save_path = os.path.join(self.out_directory, file_name)

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
    
    
class Translator():
    '''
    Initializes the Hugging Face model and defines as a function
    the pipeline to translate the segmented text afterwards.

    Once translated it recombines the datasets into a suitable format
    ---------------------
    Parameters:
      Takes no parameters
    '''
    def __init__(self,
                 en_df: pd.DataFrame,
                 es_df: pd.DataFrame
                 ):

      self.model_es_en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
      self.model_en_es = pipeline("translation", model='Helsinki-NLP/opus-mt-en-es')

      self.en_df = en_df
      self.es_df = es_df 

      self.split_es_df = None
      self.split_en_df = None

      self.trans_text_es = None
      self.trans_text_en = None

      self.translated_df_es = None
      self.translated_df_en = None

      self.tokenizer_es_en = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')
      self.tokenizer_en_es = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es')

      return
    

    def split(self, dataframe: pd.DataFrame, language: str) -> None:
      '''
      Takes an input dataset and splits its raw_text column into phrases separated by
      the dot (.)
      ----------------
      Parameters:
        dataframe: A Pandas dataframe that has been previously segmented into paragraphs. 
      '''
      def get_token_length(text):
        return len(tokenizer.encode(text, truncation=False))
      
      max_tokens  = int(512*0.9)
      #I append to lists not to pd.DataFrames
      data = []

      for i, row in dataframe.iterrows():
         
          split_text = row['text'].split(". ")

          filtered_text = filter(lambda x: x!='', split_text)

          phrases = list(filtered_text)

          #used to check max lengths
          if row['lang'] == 'es':
             tokenizer = self.tokenizer_es_en 
          else:
             tokenizer = self.tokenizer_en_es

          for _, p in enumerate(phrases):

            #Handle sentences over token limit
            if get_token_length(p) < max_tokens:
             
              new_row = [
                row["title"], row["summary"], p, row["lang"], row["url"],
                None, #For the dynamic ID
                row["equivalence"], row["id_preproc"] + "_" + str(_)
              ]

              data.append(new_row)

              valid_entry = True      

          if not valid_entry:
            if row['lang'] == 'es':
                self.es_df = self.es_df[self.es_df['id_preproc'] != row['id_preproc']]
            else:
                self.en_df = self.en_df[self.en_df['id_preproc'] != row['id_preproc']]        
              
      if language == 'en':

        en_df = pd.DataFrame(data, columns=["title", "summary", "text", "lang", "url", "index", "equivalence", "id_preproc"])
        en_df["index"] = range(len(en_df))
        self.split_en_df = en_df          

      elif language == 'es':

        es_df = pd.DataFrame(data, columns=["title", "summary", "text", "lang", "url", "index", "equivalence", "id_preproc"])
        es_df["index"] = range(len(es_df))
        self.split_es_df = es_df

      else:
         raise(Exception(f'The language: {language} is not supported!'))

      return

    def translate(self) -> None:
        '''
        Translates the dataframes to the other respective language
        using Hugging Face OPUS model
        '''       

        self.split(self.en_df, 'en')
        self.split(self.es_df, 'es')

        # Convert Pandas DataFrames to Hugging Face Datasets
        ds_en = Dataset.from_pandas(self.split_en_df)

        ds_es = Dataset.from_pandas(self.split_es_df)

        '''
        self.trans_text_en = self.split_es_df['text'].map(lambda x: self.model_es_en(x))
        
        self.trans_text_es = self.split_en_df['text'].map(lambda x: self.model_en_es(x))

       #Changes format to strings
        self.trans_text_en = self.trans_text_en.explode().apply(lambda x: x["translation_text"] if isinstance(x, dict) else None)

        self.trans_text_es = self.trans_text_es.explode().apply(lambda x: x["translation_text"] if isinstance(x, dict) else None)        
        '''
        def translate_text(batch, model):
            translation_list = model(batch['text'])
            batch['translated_text'] = [text['translation_text'] for text in translation_list]
            return batch
        
        # Apply translation using Dataset.map()
        ds_en = ds_en.map(lambda batch: translate_text(batch, self.model_en_es), batched=True)
        ds_es = ds_es.map(lambda batch: translate_text(batch, self.model_es_en), batched=True)

        self.trans_text_en = ds_es.to_pandas()['translated_text']
        self.trans_text_es = ds_en.to_pandas()['translated_text']
            
        # First update the split datasets with the translated columns
        self.split_en_df['text'] = self.trans_text_es 
        self.split_es_df['text'] = self.trans_text_en 

        self.assemble_dataframes()
        return

    
    def assemble_dataframes(self) -> None:
       '''
       Prepares the final datasets to be saved by the segmenter, 
       substitutes the new translated column, in the df, groups by 
       paragraph and concatenates to the original datasets
       '''
       #First update the split datasets with the translated columns
       self.split_en_df['text'] = self.trans_text_es 
       self.split_es_df['text'] = self.trans_text_en 

       #Now I concatenate from phrases to paragraphs

       self.split_es_df["aux_id"] = self.split_es_df["id_preproc"].str.rsplit("_", n=1).str[0]
       self.split_en_df["aux_id"] = self.split_en_df["id_preproc"].str.rsplit("_", n=1).str[0]
       
       
       grouped_es = (self.split_es_df
                     .groupby("aux_id")["text"]
                     .agg(lambda x: ' '.join(x.astype(str).str.strip()))
                     .reset_index()
                     .rename(columns={"text": "assembled_text"}))
       grouped_en = (self.split_en_df
                     .groupby("aux_id")["text"]
                     .agg(lambda x: ' '.join(x.astype(str).str.strip()))
                     .reset_index()
                     .rename(columns={"text": "assembled_text"}))
                       
       #Erase and ename columns
       merged_es_df = (self.split_es_df
                      .merge(grouped_es, on="aux_id", how="outer")
                      .drop(columns=['text', 'id_preproc'])
                      .rename(columns={'aux_id': 'id_preproc', 
                                      'assembled_text': 'text'})
                      .drop_duplicates(subset=['id_preproc'])
                      .assign(id_preproc=lambda x: 'T_' + x['id_preproc'])
                      .assign(lang=lambda x: 'en')
                      .reset_index(drop=True))
        
       merged_en_df = (self.split_en_df
                      .merge(grouped_en, on="aux_id", how="outer")
                      .drop(columns=['text', 'id_preproc'])
                      .rename(columns={'aux_id': 'id_preproc', 
                                      'assembled_text': 'text'})
                      .drop_duplicates(subset=['id_preproc'])
                      .assign(id_preproc=lambda x: 'T_' + x['id_preproc'])
                      .assign(lang=lambda x: 'es')
                      .reset_index(drop=True))
       
       #Now concatenate the translated to en dataset with the original en dataset
       self.translated_df_en = pd.concat([self.en_df, merged_es_df], ignore_index=True)
       self.translated_df_es = pd.concat([self.es_df, merged_en_df], ignore_index=True)
       

       return 


       
    def save_dataframes(self, path: str) -> None:
       date_name = str(date.today())
       
       fname_en = f'en_{date_name}_segm_trans'
       fname_es = f'es_{date_name}_segm_trans'

       save_path_en = os.path.join(path, fname_en)
       save_path_es = os.path.join(path, fname_es)

       self.translated_df_en.to_parquet(path=save_path_en, compression="gzip")
       print(f"Saving in PC: {save_path_en}")

       self.translated_df_es.to_parquet(path=save_path_es, compression="gzip")
       print(f"Saving in PC: {save_path_es}")
       return
    
class DataPreparer():
  '''
  Prepares the outputs of the NLPipe to be fed into mallet. Namely:
  Joins both datasets into one, differentiating with colum "lang",
  changes column "text" into "raw_text"
  '''
  def __init__(self,
              path_folder: str,
              name_es: str,
              name_en: str,
              final_df: pd.DataFrame = None,
              storing_path: str = ''):
    
    self.path_folder = path_folder
    self.storing_path = storing_path
    self.name_es = name_es
    self.name_en = name_en
    
    self.es_df = None
    self.en_df = None
    self.final_df = final_df if final_df is not None else pd.DataFrame(columns=['id',
                                                                                'raw_text',
                                                                                'lemmas',
                                                                                'lang'])

    return

  def read_dataframes(self) -> None:
    '''
    From the path, checks the path, checks the files
    stores them into the object as pd.Dataframes
    '''
    for df in [self.name_en, self.name_es]:
      import pdb; pdb.set_trace()
      if not os.path.exists(self.path_folder):
        raise Exception('Path not found, check again')

      elif not os.path.isfile(os.path.join(self.path_folder, df)):
        raise Exception(f'File {df} not found, check again')

      else:
        dataframe = pd.read_parquet(os.path.join(self.path_folder, df))

        if df == self.name_en:
            self.en_df = dataframe

        elif df == self.name_es:
            self.es_df = dataframe

        else:
            raise Exception(f'The name {df} does not exist in folder!')
        
        print(f"File {df} read sucessfully!")

    return

  def format_dataframes(self) -> None:
    '''
    Formats correctly the dataframes in en & es by joining them and 
    adding the correct columns

    '''
    self.read_dataframes()

    #Filter out the contaminated rows

    print('Decontaminating')
    filtered_es = self.es_df[~self.es_df['id_preproc'].str.startswith('T_ES_')]
    self.es_df = filtered_es[~filtered_es['id_preproc'].str.startswith('EN_')] 

    filtered_en = self.en_df[~self.en_df['id_preproc'].str.startswith('T_EN_')]
    self.en_df = filtered_en[~filtered_en['id_preproc'].str.startswith('ES_')]
    self.en_df = self.en_df[~self.en_df["raw_text"].str.contains("isbn", case=False, na=False)]

    print('Ordering')
    #Order both dataframes in the same fashion by aligning the spanish texts
    df_ordered = self.es_df.copy().reset_index()

    # Find the first occurrence where id_preproc starts with "T"
    idx = df_ordered[df_ordered["id_preproc"].str.startswith("T")].index.min()
    import pdb; pdb.set_trace()
    if idx is not None and not pd.isna(idx):
        # Select the portion of the DataFrame from that index onward
        df_pre = df_ordered.iloc[idx:].copy()
        df_pos = df_ordered.iloc[:idx].copy()
        # Prepend it to the original DataFrame
        df_ordered = pd.concat([df_pre, df_pos], ignore_index=True)

    self.es_df = df_ordered

    #Creating the language column
    self.en_df.loc[:, 'lang'] = 'EN'
    self.es_df.loc[:, 'lang'] = 'ES'

    min_length = min(len(self.en_df), len(self.es_df))

    # Truncate both DataFrames
    self.en_df = self.en_df.iloc[:min_length]
    self.es_df = self.es_df.iloc[:min_length]

    #delete the old id to create a new one
    if 'id' in self.en_df.columns:
        self.en_df = self.en_df.drop(columns=['id'])

    if 'id' in self.es_df.columns:
        self.es_df = self.es_df.drop(columns=['id'])

    print('Merging')
    #merge
    self.final_df = pd.concat([self.en_df, self.es_df], ignore_index=True)

    #create new index
    self.final_df['doc_id'] = self.final_df.index

    self.save_to_parquet()

    return
    
  def save_to_parquet(self) -> None:
    '''
    Saves changes ot parquet
    '''
    file_name = 'polylingual_df'

    save_path = os.path.join(self.storing_path, file_name)

    self.final_df.to_parquet(path=save_path, compression="gzip")
    print(f"Saving in PC: {save_path}")

    return