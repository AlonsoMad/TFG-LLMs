import pandas as pd
import os
import logging
from datetime import date

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

        en_df = self.segmented_df[self.segmented_df['lang'] == 'en']
        es_df = self.segmented_df[self.segmented_df['lang'] == 'es']

        self.save_to_parquet(en_df, 'en')
        self.save_to_parquet(es_df, 'es')

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

      #TODO: Check the method works!
      '''
      segmented_es_name = 'es' + self.segmented_f_name
      segmented_en_name = 'en' + self.segmented_f_name

      segmented_es_df = pd.read_parquet(os.path.join(self.segmented_path, segmented_es_name))
      segmented_en_df = pd.read_parquet(os.path.join(self.segmented_path, segmented_en_name))

      #perform the joins by id, maybe segmented_en_df.use set_index('id')
      self.en_df = self.en_df.merge(segmented_en_df, on='id', how='outer', suffixes=('', '_seg'))
      self.es_df = self.es_df.merge(segmented_es_df, on='id', how='outer', suffixes=('', '_seg'))

      #delete the old id to create a new one
      self.en_df.drop(columns=['id'])
      self.es_df.drop(columns=['id'])
      '''

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