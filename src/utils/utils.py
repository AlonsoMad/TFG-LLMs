import os
import pandas as pd

def read_data(file_path) -> pd.DataFrame: 
    '''
    Reads the specified paths for thetas and for the DataFrame
    initializes values to the og_df and thetas attributes
    '''
    if not os.path.exists(file_path):
        raise Exception('File path not found, check again')
    else:
        df = pd.read_parquet(file_path)
        print("Dataframe read sucessfully!")
    return df