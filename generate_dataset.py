import src.API.wiki_retrieve
import os

if __name__ == "__main__":
    # Define output directory
    output_dir = os.path.join(os.getcwd(), "wiki_datasets")
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
