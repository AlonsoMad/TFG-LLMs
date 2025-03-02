import requests
from concurrent.futures import ThreadPoolExecutor

def calculate_cohr(words):
    url = "http://palmetto.demos.dice-research.org/service/npmi?words="
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