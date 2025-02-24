import pandas as pd
import os
import logging
from datetime import date
from transformers import pipeline

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

        for i, row in self.input_df.iterrows():

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

          progress = 100*i/len(self.input_df)

          last_prog = 0

          if progress % 10 == 0 and progress != last_prog:

              logging.info(f"Progress: {progress}%")
              last_prog = progress            

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
    
class DataPreparer():
    '''
    Prepares the outputs of the NLPipe to be fed into Mallet. 
    This involves:
      - Merging two datasets (English & Spanish) into one, adding a "lang" column.
      - Renaming the "text" column to "raw_text".
      - Joining with segmented dataframes based on "id".
      - Saving the final processed dataframe to a parquet file.

    -------------
    Parameters:
    path_folder (str): 
        The directory where the original English and Spanish datasets are stored.
    
    segmented_path (str): 
        The directory where the segmented datasets are located.
    
    segmented_f_name (str): 
        The base filename of the segmented datasets (without language prefix).
    
    name_es (str): 
        The filename of the Spanish dataset inside `path_folder`.
    
    name_en (str): 
        The filename of the English dataset inside `path_folder`.
    
    es_df (pd.DataFrame, optional): 
        A pandas DataFrame holding the Spanish dataset. Defaults to None (will be loaded from `name_es`).
    
    en_df (pd.DataFrame, optional): 
        A pandas DataFrame holding the English dataset. Defaults to None (will be loaded from `name_en`).
    
    final_df (pd.DataFrame, optional): 
        The final merged dataset. Defaults to an empty DataFrame with predefined columns.
    
    storing_path (str, optional): 
        The directory where the processed dataset will be saved. Defaults to an empty string.
    '''
    def __init__(self,
                path_folder: str,
                segmented_path: str,
                segmented_f_name: str,
                name_es: str,
                name_en: str,
                es_df: pd.DataFrame = None,
                en_df: pd.DataFrame = None,
                final_df: pd.DataFrame = None,
                storing_path: str = ''):
      
      self.path_folder = path_folder
      self.segmented_path = segmented_path
      self.segmented_f_name = segmented_f_name
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

      #Creating the language column
      self.en_df['lang'] = 'EN'
      self.es_df['lang'] = 'ES'

      #There have been cases of spanish instances in the english df and viceversa
      if self.en_df['id'].str.startswith('ES_').any:
        self.en_df = self.en_df[~self.en_df['id'].str.startswith('ES_')]

      if self.es_df['id'].str.startswith('EN_').any:
        self.es_df = self.es_df[~self.es_df['id'].str.startswith('EN_')]

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
      self.model_en_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")

      self.en_df = en_df
      self.es_df = es_df 

      self.split_es_df = None
      self.split_en_df = None

      self.trans_text_es = None
      self.trans_text_en = None

      self.translated_df_es = None
      self.translated_df_en = None
      return

    def split(self, dataframe: pd.DataFrame, language: str) -> None:
      '''
      Takes an input dataset and splits its raw_text column into phrases separated by
      the dot (.)
      ----------------
      Parameters:
        dataframe: A Pandas dataframe that has been previously segmented into paragraphs. 
      '''

      for i, row in dataframe.iterrows():
         
          split_text = row['text'].split(". ")

          filtered_text = filter(lambda x: x!='', split_text)

          phrases = list(filtered_text)

          for _, p in enumerate(phrases):
             
            if dataframe['lang'] == 'es':

                self.split_es_df.loc[len(self.split_es_df)] = [row['title'],
                                  row['summary'],
                                  p,
                                  row['lang'],
                                  row['url'],
                                  len(self.split_es_df),
                                  row["equivalence"],
                                  row['id_preproc']+"_"+str(_)]
            else:
               
                self.split_en_df.loc[len(self.split_en_df)] = [row['title'],
                                  row['summary'],
                                  p,
                                  row['lang'],
                                  row['url'],
                                  len(self.split_en_df),
                                  row["equivalence"],
                                  row['id_preproc']+"_"+str(_)]   
            
      return

    def translate(self) -> None:
       '''
       Translates the dataframes to the other respective language
       using hugginface OPUS model
       '''       
       self.trans_text_en = self.split_es_df['text'].map(lambda x: self.model_es_en(x))
        
       self.trans_text_es = self.split_en_df['text'].map(lambda x: self.model_en_es(x))

      #Changes format to strings
       self.trans_text_en = self.trans_text_en.explode().apply(lambda x: x["translation_text"] if isinstance(x, dict) else None)

       self.trans_text_es = self.trans_text_es.explode().apply(lambda x: x["translation_text"] if isinstance(x, dict) else None)
         
       return
    
    def assemble_dataframes(self) -> None:
       '''
       Prepares the final datasets to be saved by the segmenter, 
       substitutes the new translated column, in the df, groups by 
       paragraph and concatenates to the original datasets
       '''
       #First update the split datasets with the translated columns
       self.split_es_df['text'] = self.trans_text_es 
       self.split_en_df['text'] = self.trans_text_en 

       #Now I concatenate from phrases to paragraphs

       self.split_es_df["aux_id"] = self.split_es_df["id"].str.rsplit("_", n=1).str[0]

        # Group by parent_id and concatenate children's text
       self.split_es_df = self.split_es_df.groupby("aux_id")["text"].apply(lambda x: " ".join(x)).reset_index()

       return




    def save(self) -> None:
       pass
    

    