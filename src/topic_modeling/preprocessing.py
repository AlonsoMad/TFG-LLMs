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