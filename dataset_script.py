'''
31-Mar:
Este script en un principio descargaría el dataset elegido para el proceso de testeo del Q&A.
Si en un futuro su funcionalidad cambiara o fuese expandida, lo anotaría aquí.
'''
import pandas as pd
import json
import pdb


# Generate the dataset
def main():
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
        'question': df['question']
    })

    results.to_parquet(save_path_df, compression='gzip')
    questions.to_parquet(save_path_question, compression='gzip')

    return

if __name__ == '__main__':
    main()