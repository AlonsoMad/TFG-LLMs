import pandas as pd
import numpy as np
import scipy.stats as stats
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import ast

def extract_doc_ids(row, key):
    try:
        if isinstance(row[key], np.ndarray):
            row[key] = row[key].tolist() 
        if isinstance(row[key], str):
            row_ = ast.literal_eval(row[key])
            return [el["doc_id"] for el in row_[0]]
        if isinstance(row[key], list) and len(row[key]) > 0:
            return [el["doc_id"] for el in row[key][0]] 
    except:
        flattened_list = [{'doc_id': entry['doc_id'], 'score': entry['score']} for subarray in row[key] for entry in subarray]
        return [el["doc_id"] for el in flattened_list]
        
    return []
    
def precision_at_k(row, key, k=3):
    retrieved_docs = set(row[f"doc_ids_{key}"][:k])  
    relevant_docs = set(row["relevant_docs"])

    if not retrieved_docs:
        return 0.0

    return len(retrieved_docs & relevant_docs) / len(retrieved_docs)

def recall_at_k(row, key, k=3):
    retrieved_docs = set(row[f"doc_ids_{key}"][:k])
    relevant_docs = set(row["relevant_docs"])

    if not relevant_docs:
        return 0.0

    return len(retrieved_docs & relevant_docs) / len(relevant_docs)

def multiple_mean_reciprocal_rank_at_k(row, key, k):
    retrieved_docs = row[f"doc_ids_{key}"][:k]
    relevant_docs = set(row["relevant_docs"])
    
    ranks = []
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            ranks.append(i + 1)
    
    if not ranks:
        return 0

    numerator = np.mean(ranks)

    n = len(relevant_docs)
    denominator = (n / 2) + ((k + 1) * (n - len(ranks)))

    return numerator / denominator

def dcg_at_k(scores, k):
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(scores[:k]))

def ndcg_at_k(row, key, k=10):
    retrieved_docs = row[f"doc_ids_{key}"][:k]
    relevant_docs = set(row["relevant_docs"])
    
    # 1 if the document is relevant, 0 otherwise
    gains = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]
    
    dcg = dcg_at_k(gains, k)
    
    ideal_gains = sorted(gains, reverse=True)
    idcg = dcg_at_k(ideal_gains, k)
    
    return dcg / idcg if idcg > 0 else 0

########
k = 3
tpc = 11
########
# Paths to Excel files
excel_paths = [  
    (f"GENERATIONS/outs_good_model_tpc{tpc}/relevant_check/questions_topic_{tpc}_qwen:32b_100_seed_1234_results_model30tpc_thr__combined_to_retrieve_relevant.parquet",
        [
            f"GENERATIONS/outs_good_model_tpc{tpc}/relevant/questions_topic_{tpc}_qwen:32b_100_seed_1234_results_model30tpc_thr_.parquet",
            f"GENERATIONS/outs_good_model_tpc{tpc}/relevant/questions_topic_{tpc}_qwen:32b_100_seed_1234_results_model30tpc_thr__dynamic.parquet"
        ]    
    ),
    (f"GENERATIONS/outs_good_model_tpc{tpc}/relevant_check/questions_topic_{tpc}_llama3.3:70b_100_seed_1234_results_model30tpc_thr__combined_to_retrieve_relevant.parquet",
        [
            f"GENERATIONS/outs_good_model_tpc{tpc}/relevant/questions_topic_{tpc}_llama3.3:70b_100_seed_1234_results_model30tpc_thr_.parquet",
            f"GENERATIONS/outs_good_model_tpc{tpc}/relevant/questions_topic_{tpc}_llama3.3:70b_100_seed_1234_results_model30tpc_thr_.parquet",
        ]    
    ),
    (f"GENERATIONS/outs_good_model_tpc{tpc}/relevant_check/questions_topic_{tpc}_gpt-4o-2024-08-06_100_seed_1234_results_model30tpc_thr__combined_to_retrieve_relevant.parquet",
        [
            f"GENERATIONS/outs_good_model_tpc{tpc}/relevant/questions_topic_{tpc}_gpt-4o-2024-08-06_100_seed_1234_results_model30tpc_thr_.parquet",
            f"GENERATIONS/outs_good_model_tpc{tpc}/relevant/questions_topic_{tpc}_gpt-4o-2024-08-06_100_seed_1234_results_model30tpc_thr__dynamic.parquet"]
    ),
]

# Ensure method mapping is correctly defined
method_mapping = {
        "1": "ENN",
        "2": "ANN",
        "3_weighted": "TB-ENN-W",
        "3_unweighted": "TB-ENN",
        "4_weighted": "TB-ANN-W",
        "4_unweighted": "TB-ANN",
        "time_1": "ENN",
        "time_2": "ANN",
        "time_3_weighted": "TB-ENN-W",
        "time_3_unweighted": "TB-ENN",
        "time_4_weighted": "TB-ANN-W",
        "time_4_unweighted": "TB-ANN",
        
}

def extract_metric_method(metric):
    for key, value in method_mapping.items():
        if metric.endswith(f"_{key}") or f"_{key}_" in metric:
            base_metric = (
                #"MRR" if "mrr_" in metric else
                "MRR@3" if "mrr_3" in metric else
                "MRR@5" if "mrr_5" in metric else
                "MRR@10" if "mrr_10" in metric else
                "NDCG@3" if "ndcg_3" in metric else
                "NDCG@5" if "ndcg_5" in metric else
                "NDCG@10" if "ndcg_10" in metric else
                "Precision@3" if "precision_3" in metric else
                "Precision@5" if "precision_5" in metric else
                "Precision@10" if "precision_10" in metric else
                "Recall@3" if "recall_3" in metric else
                "Recall@5" if "recall_5" in metric else
                "Recall@10" if "recall_10" in metric else
                "Time (s)" if "time" in metric else
                None
            )
            return base_metric, value  # Return (metric type, merged method)
    return None, metric  # Fallback

# Extract the number of relevant passages for each query
#annotations_grouped["num_relevant_docs"] = annotations_grouped["relevant_docs"].apply(lambda x: len(x) if isinstance(x, list) else 0)


#DE aquí a abajo están los cálculos de métricas

# Define the retrieval metrics to compute weighted means for
retrieval_metrics = [
    "mrr_3", "mrr_5",
    #"mrr_10", 
    "precision_3", "precision_5", 
    #"precision_10",
    "recall_3", "recall_5", 
    #"recall_10", 
    "ndcg_3", "ndcg_5", 
    #"ndcg_10"
]

# Read and process results from all files
methods = ["QWEN", "LLAMA", "GPT"] 
for id_method, method_tuple in enumerate(excel_paths):
    print(f"\n***** {methods[id_method]} *****\n")

    annotations_keep = pd.read_parquet(method_tuple[0].replace(".parquet", "_all_added.parquet"))
    #annotations_keep = pd.read_parquet(method_tuple[0].replace(".parquet", "_qwen_added.parquet"))
    annotations_keep = annotations_keep[(annotations_keep["relevance_llama3.3:70b"] == 1) & (annotations_keep["relevance_qwen:32b"] == 1) & (annotations_keep["relevance_llama3:70b-instruct"] == 1) & (annotations_keep["relevance_llama3.1:8b-instruct-q8_0"] == 1)]# 
    annotations_grouped = annotations_keep.groupby('question')['all_results'].apply(list).reset_index()
    annotations_grouped = annotations_grouped.rename(columns = {"all_results": "relevant_docs"})
    annotations_grouped["num_relevant_docs"] = annotations_grouped["relevant_docs"].apply(lambda x: len(x))
    print(f"Number of relevant passages for {methods[id_method]}: {annotations_grouped['num_relevant_docs'].mean():.3f}±{annotations_grouped['num_relevant_docs'].std():.3f}")

    weighted_results = []
    for file_idx, path_results in enumerate(method_tuple[1]):
        df_results = pd.read_parquet(path_results)
        df_results = df_results.drop_duplicates(subset=['question'], keep='first')
        df_results = df_results.rename(columns={"questions": "question"})
        df_results = df_results.merge(annotations_grouped, on="question")
    
        for key in ["results_1", "results_2", "results_3_weighted", "results_3_unweighted",
                    "results_4_weighted", "results_4_unweighted"]:
            
            df_results[f"doc_ids_{key}"] = df_results.apply(lambda x: extract_doc_ids(x, key), axis=1)
            for k in [3,5,10]:
                #import pdb; pdb.set_trace()
                df_results[f"mrr_{k}_{key}"] = df_results.apply(lambda x: multiple_mean_reciprocal_rank_at_k(x, key, k=k), axis=1)
                df_results[f"precision_{k}_{key}"] = df_results.apply(lambda x: precision_at_k(x, key, k=k), axis=1)
                df_results[f"recall_{k}_{key}"] = df_results.apply(lambda x: recall_at_k(x, key, k=k), axis=1)
                df_results[f"ndcg_{k}_{key}"] = df_results.apply(lambda x: ndcg_at_k(x, key, k=k), axis=1)

            df_results["time_3_weighted"] = df_results["time_3"]
            df_results["time_4_weighted"] = df_results["time_4"]
            df_results["time_3_unweighted"] = df_results["time_3"]
            df_results["time_4_unweighted"] = df_results["time_4"]
            
            method_name = method_mapping.get(key, key)

            print(f"Calculating for retrieval metrics: {retrieval_metrics}")
            for metric in retrieval_metrics:
                metric_column = f"{metric}_{key}"
    
                if metric_column in df_results.columns:
                    values = df_results[metric_column].dropna()
                    weights = df_results.loc[values.index, "num_relevant_docs"].fillna(1)
    
                    weighted_mean = np.average(values, weights=weights)
                    weighted_var = np.average((values - weighted_mean) ** 2, weights=weights)
                    weighted_std = np.sqrt(weighted_var)
                    n = len(values)
                    ci = 1.96 * (weighted_std / np.sqrt(n)) if n > 1 else 0
    
                    metric_name, method_display = extract_metric_method(metric_column)
                    weighted_results.append((file_idx, method_display, metric_name, weighted_mean, ci))

            print("Calcualting for time")
            # do the same for the time
            metric = "time"
            key_for_time = key.split("results_")[-1]
            #print(key_for_time)
            metric_column = f"{metric}_{key_for_time}"
            #print(metric_column)
            values = df_results[metric_column].dropna()
            mean_values = values.mean()
            std_values = values.std()
            metric_name, method_display = extract_metric_method(metric_column)
            #print(f"Key for time {metric_name}, {method_display}")
            weighted_results.append((file_idx, method_display, metric_name, mean_values, std_values))
    
    # Convert results into a structured DataFrame
    df_weighted = pd.DataFrame(weighted_results, columns=["FileID", "Method", "Metric", "Weighted Mean", "95% CI"])
    
    df_weighted["BestValue"] = df_weighted.groupby(["FileID", "Metric"])["Weighted Mean"].transform(
        lambda x: x.min() if "Time (s)" in x.name else x.max()
    )
    
    """
    def format_weighted_value(row):
        mean = f"{row['Weighted Mean']:.3f}"
        ci = f"{row['95% CI']:.3f}"
        value = f"{mean} ± {ci}"
        if row["Weighted Mean"] == row["BestValue"]:
            value = f"\\textbf{{{value}}}"
        return value
    #df_weighted["Weighted Mean ± CI"] = df_weighted.apply(format_weighted_value, axis=1)
    """
    
    #### SIGNIFICANCE TESTS #####
    # Define function to add significance markers based on p-values
    def get_significance_marker(p_value):
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return ""
    
    # Modify format function to add significance markers
    def format_weighted_value(row):
        mean = f"{row['Weighted Mean']:.3f}"
        ci = f"{row['95% CI']:.3f}"
        value = f"{mean} \pm {ci}"
    
        metric = row["Metric"]
        method = row["Method"]
    
        """
        # Agregar '*' si Kruskal-Wallis indica diferencias significativas
        if metric in significance_dict["Kruskal-Wallis p-value"]:
            kw_p_value = significance_dict["Kruskal-Wallis p-value"][metric]
            if kw_p_value < 0.05:
                value += " *"  # Indica que hay diferencias globales
        """
        significance_marker = ""
        if metric in significance_dict["Significant post-hoc tests (Dunn)"]:
            dunn_df = significance_dict["Significant post-hoc tests (Dunn)"][metric]
            for other_method in dunn_df.index:
                if method in dunn_df.columns and other_method in dunn_df.index:
                    p_val = dunn_df.loc[other_method, method]
                    if p_val < 0.001:
                        significance_marker = "***"
                    elif p_val < 0.01:
                        significance_marker = "**"
                    elif p_val < 0.05:
                        significance_marker = "*"
    
        value += f"^{{{significance_marker}}}"
    
        if row["Weighted Mean"] == row["BestValue"]:
            return f"\\(\\boldsymbol{{{value}}}\\)"
        else:
            return f"\\({value}\\)"
            
    def format_metric(metric):
        return (
                "MRR@3" if metric == "mrr_3" else
                "MRR@5" if metric == "mrr_5" else
                "NDCG@3" if metric == "ndcg_3" else
                "NDCG@5" if metric == "ndcg_5" else
                "Precision@3" if metric == "precision_3" else
                "Precision@5" if metric == "precision_5" else
                "Recall@3" if metric == "recall_3" else
                "Recall@5" if metric == "recall_5" else
                "Time (s)" if "time" in metric else
                metric  # Fallback si no está en la lista
            )
    
    #### TO CALCULATE SIGNIFICANCE TESTS
    df_columns_keep = [col for col in df_results.columns if any(metric in col for metric in retrieval_metrics + ["time"])]
    #print(df_columns_keep)
    
    metric_columns = [col for col in df_results[df_columns_keep].columns if ("results_" in col or "time" in col) and ("time_3" not in col and "time_4" not in col)]
    df_long = df_results.melt(value_vars=metric_columns, var_name="Metric", value_name="Value")
    
    df_long["Method"] = df_long["Metric"].apply(lambda x: method_mapping[x.split("_results_")[-1]])
    df_long["Metric"] = df_long["Metric"].apply(lambda x: format_metric(x.split("_results_")[0]))
    
    # Check normality
    normality_tests = {}
    for metric in df_long["Metric"].unique():
        for method in df_long["Method"].unique():
            values = df_long[(df_long["Metric"] == metric) & (df_long["Method"] == method)]["Value"]
            if len(values) > 3:  # Shapiro necesita al menos unos pocos valores
                stat, p_value = stats.shapiro(values)
                normality_tests[(metric, method)] = p_value
    
    # ANOVA or Kruskal-Wallis
    anova_results = {}
    kruskal_results = {}
    
    for metric in df_long["Metric"].unique():
        groups = [df_long[(df_long["Metric"] == metric) & (df_long["Method"] == method)]["Value"].dropna() for method in df_long["Method"].unique()]
        # if normality in at least one group
        normal = all(p > 0.05 for p in [normality_tests.get((metric, method), 0) for method in df_long["Method"].unique()])
        
        if normal:
            stat, p_value = stats.f_oneway(*groups)
            anova_results[metric] = p_value
        else:
            stat, p_value = stats.kruskal(*groups)
            kruskal_results[metric] = p_value
    
    # post-hoc tests if Kruskal-Wallis is significant
    posthoc_results = {}
    for metric, p in kruskal_results.items():
        if p < 0.05:  # Si hay diferencias significativas
            df_filtered = df_long[df_long["Metric"] == metric]
            posthoc = sp.posthoc_dunn(df_filtered, val_col="Value", group_col="Method", p_adjust="bonferroni")
            posthoc_results[metric] = posthoc
    
    significance_dict = {
        "ANOVA p-value": anova_results,
        "Kruskal-Wallis p-value": kruskal_results,
        "Significant post-hoc tests (Dunn)": posthoc_results
    }
    
    df_weighted["Weighted Mean ± CI"] = df_weighted.apply(format_weighted_value, axis=1)
    
    df_weighted_pivot = df_weighted.pivot(index=["FileID", "Method"], columns="Metric", values="Weighted Mean ± CI")
    
    latex_output = "\\begin{table*}[h]\n\\centering\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{lc" + "c" * len(df_weighted_pivot.columns) + "}\n\\arrayrulecolor{black}\n\\toprule\n"
    
    latex_output += "File & \\textbf{Method} & " + " & ".join([f"\\textbf{{{col}}}" for col in df_weighted_pivot.columns]) + " \\\\\n\\midrule\n"
    
    current_file_id = None 
    for (file_idx, method), row in df_weighted_pivot.iterrows():
        if file_idx != current_file_id:  
            if current_file_id is not None:
                latex_output += "\\arrayrulecolor{black}\\specialrule{0.5pt}{0pt}{0pt}\\arrayrulecolor{black}\n"
            
            latex_output += f"\\multicolumn{{{len(df_weighted_pivot.columns) + 2}}}{{l}}{{\\textbf{{File {file_idx+1}}}}} \\\\\n"
            
            latex_output += "\\arrayrulecolor{black}\\specialrule{0.5pt}{0pt}{0pt}\\arrayrulecolor{black}\n"
            
            current_file_id = file_idx  
        
        row_start = "\\rowcolor{tableblue} " if method == "XX" else ""
        
        latex_output += row_start + f"& \\textbf{{{method}}} & " + " & ".join(row.astype(str)) + " \\\\\n"
    
    latex_output += "\\bottomrule\n\\end{tabular}}\n\\caption{Performance Metrics per Method for Each File (Best values in bold)}\n\\label{tab:grouped_results}\n\\end{table*}"
    
    print(latex_output)