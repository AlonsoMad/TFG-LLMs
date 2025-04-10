import pandas as pd # type: ignore
from prompter import Prompter

template_path = "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/templates/test_relevance.txt"
with open(template_path, 'r') as file:
    template = file.read()

paths_ = [
    "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/GENERATIONS/outs_good_model_tpc24/relevant/questions_topic_24_qwen:32b_100_results_model30tpc_thr__combined_to_retrieve_relevant.parquet",
    "/export/usuarios_ml4ds/lbartolome/Repos/umd/LinQAForge/src/qa_system/GENERATIONS/outs_good_model_tpc11/relevant/questions_topic_11_qwen2.5:32b_100_seed_1234_results_model30tpc_thr__combined_to_retrieve_relevant.parquet",
    
]

for path_annotations in paths_:
    annotations = pd.read_parquet(path_annotations)
    for llm_model in ["qwen:32b", "llama3.3:70b", "llama3:70b-instruct", "llama3.1:8b-instruct-q8_0"]:
        
        prompter = Prompter(
            model_type=llm_model,
        )
        print(f"Processing for LLM {llm_model}")
        annotations[f"relevance_{llm_model}"] = len(annotations) * [None]
        for id_row, row in annotations.iterrows():
            question = template.format(passage=row.all_results_content, question=row.question)
            response, _ = prompter.prompt(
                #system_prompt_template_path=system_template_path,
                question=question
            )
            relevance = 1 if "yes" in response.lower() else 0
            annotations.loc[id_row, f"relevance_{llm_model}"] = relevance
        
            if id_row % 100 == 0: 
                print(f"Processed {len(annotations) - id_row} / {len(annotations)} using LLM {llm_model}")
                
    try:        
        annotations.to_parquet(path_annotations.replace("/relevant/", "/relevant_check/"), index=False)
    except Exception as e:
        print(f"Error saving annotations for {path_annotations}. Error: {e}")
        import pdb; pdb.set_trace()