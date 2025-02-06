from src.API.wiki_retrieve import *
import os

if __name__ == "__main__":
    # Define output directory
    dir = os.path.dirname(os.path.abspath(__file__))
    father_dir = os.path.split(dir)[0]

    output_dir = os.path.join(father_dir, "DATA")

    os.makedirs(output_dir, exist_ok=True)

    # Initialize WikiRetriever with desired parameters
    retriever = WikiRetriever(
        file_path=output_dir,
        seed_lan="en",
        seed_query="George Washington",
        ndocs=2000  
    )

    # Run the retrieval process
    print("Starting Wikipedia retrieval...")
    retriever.retrieval()

    # Save the dataset
    retriever.df_to_parquet()

    print(f"Dataset saved in {output_dir}/dataset.parquet.gzip")
