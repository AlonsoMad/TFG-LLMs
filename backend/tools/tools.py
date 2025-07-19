import dotenv
import os
import pandas as pd
import numpy as np

def load_datasets(dataset_path: str) -> tuple:

    dataset_list = []

    datasets_name = os.listdir(dataset_path)
    shapes = np.empty((len(datasets_name), 2), dtype=int)
    print(f"Datasets found: {datasets_name}")
    for i, d in enumerate(datasets_name):
        #For each one of the datasets load in memory a short header
        #Semi harcoded, TODO: solve in future
        ds = pd.read_parquet(os.path.join(dataset_path, d, 'polylingual_df'))
        shapes[i] = ds.shape
        ds = ds.drop(columns=['index'])
        dataset_list.append(ds.head(20))
        print(f"Dataset {d} loaded with shape {ds.shape}")
    
    return dataset_list, datasets_name, shapes