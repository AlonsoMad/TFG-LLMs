'''
31-Mar:
Este script en un principio descargaría el dataset elegido para el proceso de testeo del Q&A.
Si en un futuro su funcionalidad cambiara o fuese expandida, lo anotaría aquí.
'''
import pandas as pd
import json
import pdb
from src.mind.query_generator import QueryGenerator


# Generate the dataset
def main(seed: int = 42, n_samples: int = 10000):
    df = pd.read_parquet("hf://datasets/qiaojin/PubMedQA/pqa_artificial/train-00000-of-00001.parquet")

    save_path_df = '/export/usuarios_ml4ds/ammesa/Data/1_segmented_data/dataset_PubMedQA' 
    save_path_question = '/export/usuarios_ml4ds/ammesa/Data/question_bank/questions_PubMedQA' 

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

    questions_sampled = questions.sample(n=n_samples, random_state=seed)

    def generate_subqueries(row):
        return qg.generate_query(row['question'], row['passage'])

    questions_sampled['subqueries'] = questions_sampled.apply(generate_subqueries, axis=1)

    results.to_parquet(save_path_df, compression='gzip')
    questions_sampled.to_parquet(save_path_question, compression='gzip')

    return

if __name__ == '__main__':
    main()