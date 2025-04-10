'''
31-Mar:
Este script en un principio descargaría el dataset elegido para el proceso de testeo del Q&A.
Si en un futuro su funcionalidad cambiara o fuese expandida, lo anotaría aquí.
'''
import pandas as pd
import json
import pdb
import os
from src.mind.query_generator import QueryGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed


# Generate the dataset
def main(seed: int = 42, ncpus: int = 4):
    df = pd.read_parquet("hf://datasets/qiaojin/PubMedQA/pqa_artificial/train-00000-of-00001.parquet")

    save_path_df = '/export/usuarios_ml4ds/ammesa/Data/1_segmented_data/dataset_PubMedQA' 
    save_path_question = '/export/usuarios_ml4ds/ammesa/Data/question_bank/questions_PubMedQA_full' 

    equivalence = 0
    url = 'NOURL'
    lang = 'en'
    title = 'title'

    df['context'] = df['context'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    df['summary'] = df['context'].apply(lambda x: " ".join(x.get('contexts', [])))
    df['text'] = df['context'].apply(lambda x: " ".join(x.get('contexts', [])))
    results = pd.DataFrame({
        'title': title,
        'summary': df['summary'],
        'text': df['text'],
        'lang': lang,
        'url': url,
        'equivalence': equivalence,
        'id': df['pubid'],
        'id_preproc': df['pubid']
    })
    questions = pd.DataFrame({ 
        #TODO: Es posible que incluir el relevant docs = ID sea necesario
        'id': df['pubid'],
        'id_preproc': df['pubid'],
        'full_doc': df['text'],
        'passage': df['text'],
        'question': df['question'],
        'relevant_docs': df['pubid']
    })

    qg = QueryGenerator()
    
    questions_sampled = questions#.sample(n=n_samples, random_state=seed)

    def generate_subqueries(row):
        return qg.generate_query(row['question'], row['passage'])
    
    def process_row(row):
        row['subqueries'] = generate_subqueries(row)
        return row

    def parallel_apply(df, max_workers=20):
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_row, row) for _, row in df.iterrows()]
            for future in as_completed(futures):
                results.append(future.result())
        return pd.DataFrame(results)
    #questions_sampled['subqueries'] = questions_sampled.apply(generate_subqueries, axis=1)

    subqueried_q = parallel_apply(questions_sampled, ncpus)
    results.to_parquet(save_path_df, compression='gzip')
    subqueried_q.to_parquet(save_path_question, compression='gzip')

    return

if __name__ == '__main__':
    main()