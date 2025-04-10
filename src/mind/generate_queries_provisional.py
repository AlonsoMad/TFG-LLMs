# script to just generate queries and text retrieval per-topic and full.import pathlib
import pathlib
import re
import numpy as np
import pandas as pd
from scipy import sparse # type: ignore
from prompter import Prompter

def get_doc_top_tpcs(doc_distr, topn=10):
    # @AlonsoMad: te dejo esta función aquí para que veas que es la misma, pero no la dejaría aquí, sino que la movería a utils (fíjate que ahora mismo tienes una función con este nombre en indexado.py, retrieval.py e indexer.py)
    sorted_tpc_indices = np.argsort(doc_distr)[::-1]
    top = sorted_tpc_indices[:topn].tolist()
    return [(k, doc_distr[k]) for k in top if doc_distr[k] > 0]

def extend_to_full_sentence(
    text: str,
    num_words: int
) -> str:
    """Truncate text to a certain number of words and extend to the end of the sentence so it's not cut off.
    """
    # @AlonsoMad: Mover a utils también. Probar con diferente num_words y sin truncar directamente
    text_in_words = text.split()
    truncated_text = " ".join(text_in_words[:num_words])
    
    # Check if there's a period after the truncated text
    remaining_text = " ".join(text_in_words[num_words:])
    period_index = remaining_text.find(".")
    
    # If period, extend the truncated text to the end of the sentence
    if period_index != -1:
        extended_text = f"{truncated_text} {remaining_text[:period_index + 1]}"
    else:
        extended_text = truncated_text
    
    # Clean up screwed up punctuations        
    extended_text = re.sub(r'\s([?.!,"])', r'\1', extended_text)
    
    return extended_text

######################
# PATHS TO TEMPLATES #
######################
# 1. QUESTION GENERATION
_1_INSTRUCTIONS_PATH = "templates/question_generation.txt" 
# 2. SEARCH QUERY GENERATION
_2_INSTRUCTIONS_PATH = "templates/query_generation.txt" 

################
# ENGLISH DATA #
################
# @AlonsoMad: adapt this to your paths
PATH_SOURCE = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/source/corpus_rosie/passages/26_jan/df_1.parquet")
PATH_MODEL = pathlib.Path("/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/data/models/26_jan_no_dup/poly_rosie_1_30")
print("-- Testing query from English corpus...")

thetas_en = sparse.load_npz(PATH_MODEL / "mallet_output" / "thetas_EN.npz").toarray()
raw = pd.read_parquet(PATH_SOURCE)
raw_en = raw[raw.doc_id.str.contains("EN")].copy()
raw_en["thetas"] = list(thetas_en)
raw_en["top_k"] = raw_en["thetas"].apply(lambda x: get_doc_top_tpcs(x, topn=10))
raw_en["main_topic_thetas"] = raw_en["thetas"].apply(lambda x: np.argmax(x))

print(f"English corpus loaded with {len(raw_en)} documents.")
print(raw_en.head())

for llm_model in ["qwen:32b","llama3.3:70b","gpt-4o-2024-08-06"]: # @AlonsoMad: esto esta así porque experimentaba con varios modelos, pero la idea es que todo esto tendría que verse como el QueryGenerator, esto es, le pasas un model_type que llama al prompter, etc.
    prompter = Prompter(
        model_type=llm_model,
        ollama_host="http://kumo01.tsc.uc3m.es:11434",
    )
    
    print(f"--- Testing model {llm_model} ---")

    info_topic = []
    for topic in [15]: # @ TODO: this should be given as parameter
        print(f"Topic {topic}")
        for pass_id, pass_row in raw_en[raw_en.main_topic_thetas == topic].iterrows():
            
            with open(_1_INSTRUCTIONS_PATH, 'r') as file: template = file.read()
            
            question = template.format(passage=pass_row.text, full_document=(extend_to_full_sentence(pass_row.full_doc, 100)+ " [...]"))
            
            questions, _ = prompter.prompt(question=question)
            
            if "N/A" in questions:
                reason = questions
                questions = ""
            else:
                try:
                    for sep in ["\n", ","]:
                        if sep in questions:
                            questions_text = questions.split(sep)  # Use the detected separator
                            questions = [q.strip() for q in questions_text if q.strip() and "passage" not in q]
                            reason = ""
                            break
                    else:
                        reason = "No valid separator found"
                except Exception as e:
                    print(e)
                    questions = ""
                    reason = str(e)  
                
            info_topic.append({
                "pass_id": pass_id,
                "doc_id": pass_row.doc_id,
                "passage": pass_row.text,
                "full_doc": pass_row.full_doc,
                "top_k": pass_row.top_k,
                "questions": questions,
                "reason": reason
            })
        df_info_topic = pd.DataFrame(info_topic)

        # Create subdataFrame keeping the rows whose questions are not empty and transform it into a new DataFrame where each question is a new row, keeping the original pass_id and passage. We create a new column for the question_id
        df_info_topic = df_info_topic[df_info_topic.questions.apply(lambda x: len(x) > 0)]
        df_info_topic = df_info_topic.explode('questions').reset_index(drop=True)
        df_info_topic = df_info_topic.rename(columns={"questions": "question"})
        df_info_topic['question_id'] = df_info_topic.index
        
        df_info_topic["queries"] = None
        
        for question_id, question_row in df_info_topic.iterrows():
            
            print(f"--- Processing QUESTION {question_id} ---")
            print(f"Question: {question_row.question}") 
            
            with open(_2_INSTRUCTIONS_PATH, 'r') as file: template = file.read()
            
            question = template.format(question=question_row.question,passage=question_row.passage)
            queries, _ = prompter.prompt(question=question)#system_prompt_template_path=_2_SYSTEM_PATH,
            # extract the query
            try:
                queries_clean = [el for el in queries.split(";") if el.strip()]
            except Exception as e:
                print(f"******Error extracting queries: {e}")
            
            df_info_topic.loc[question_id, 'queries'] = str(queries_clean)
        df_info_topic.to_excel(f"GENERATIONS/OLD_MODEL/questions_tpc{topic}_{llm_model}_sample100.xlsx")