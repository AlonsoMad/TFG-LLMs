import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import numpy as np



def calculate_cohr(words):
    url = "http://palmetto.demos.dice-research.org/service/cv?words="
    data = {"words": words}
    
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        return float(response.text)
    else:
        print("Error:", response.status_code, response.text)
        return None
    
def calculate_cohr_parallel(topic_keys):
    """Calculate coherence scores in parallel for a list of topic keys."""
    with ThreadPoolExecutor() as executor:
        # Map the `calculate_cohr` function to each topic key
        coherence_scores = list(executor.map(calculate_cohr, topic_keys))
    return coherence_scores

def extract_cohr(lang:str) -> np.float64:
    path_keys = f"/export/usuarios_ml4ds/ammesa/mallet_folder/mallet_output/keys_{lang}.txt"
    with open(path_keys, 'r') as file:
        lines = file.readlines()
    topic_keys = [" ".join(line.strip().split()[:10]) for line in lines]
    cohr_per_tpc = calculate_cohr_parallel(topic_keys)
    print(f"Coherence per topic: {cohr_per_tpc}")
    avg_cohr = np.mean([c for c in cohr_per_tpc if c is not None])  
    print(f"Average coherence: {avg_cohr}")

    return avg_cohr