"""
Author: Amr Sherif
Version: 3.0.0
Date: 2024-09-24
Project: Arxiv Assistant

Description: This script processes local or remote PDF files (stored in a Google Cloud Storage (GCS) bucket) into text chunks,
             applies node-level semantic processing, and uploads the processed data to a Hugging Face dataset for further processing, curation, training or analysis.

Key functionalities:
    - Uses multiprocessing to handle large datasets more efficiently.
    - Supports configurable dataset chunk sizes for optimized memory and performance.
    - Allows flexible file processing limits per batch and folder to accommodate for different use cases and to prevent overwhelming system resources.
    - Stores checkpoints to resume processing from previously processed files in case of interruptions.
    - Integrates with Hugging Face for uploading processed datasets and GCS for retrieving the files.

Dependencies:
    - Hugging Face Hub for model embeddings.
    - LlamaIndex for document parsing and node extraction.
    - Google Cloud Storage for file retrieval.
    - PyMuPDF (fitz) for PDF parsing.
    - LangChain for embedding models.
    - dotenv for managing environment variables (tokens, credentials).
    - Datasets library for managing large datasets.

License: MIT License

Change Log:
    - Version 1.0.0: Initial version, basic PDF file processing.
    - Version 2.0.0: Added multiprocessing support to speed up file handling.
    - Version 3.0.0: Added support for chunking datasets by file size and limiting the number of files processed per batch.
"""

import os
import argparse
import multiprocessing
import shutil
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.gcs import GCSReader
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder
from huggingface_hub import notebook_login, login
from concurrent.futures import ProcessPoolExecutor, as_completed
from google.colab import userdata
import fitz  # PyMuPDF
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from dotenv import load_dotenv
from google.cloud import storage
from multiprocessing import Lock

load_dotenv()

hfToken = os.getenv('hf')
login(hfToken, add_to_git_credential=True)

parser = argparse.ArgumentParser(description="Process local PDF files into text chunks and "
                                             "upload to a Hugging Face dataset.")
parser.add_argument("--bucket_name", type=str, required=True, default="arxiv-dataset", help="Name of the GCS bucket.")
parser.add_argument("--prefix", type=str, default="arxiv-dataset/arxiv/arxiv/pdf",
                    help="Prefix to filter files in the GCS bucket.")
parser.add_argument("--model_name", type=str, help="Name of the Hugging Face model to use for embeddings.")
parser.add_argument("--local_path", type=str, default=os.getcwd(),
                    help="Local path to temporary save the dataset chunks. Default current working directory.")
parser.add_argument("--chunk_size_gb", type=float, default=40.0,
                    help="Size of each dataset chunk in GB. Default is 100 GB.")
parser.add_argument("--hf_username", type=str, help="Hugging Face username.")
parser.add_argument("--hf_dataset_name", type=str, help="Hugging Face dataset name.")
parser.add_argument("--checkpoint_file", type=str, default="checkpoint.json",
                    help="Path to the checkpoint file. Default is 'checkpoint.json'.")
parser.add_argument("--base_chunk_count", type=int, default=0,
                    help="Base chunk count to start from. Default is 0.")
parser.add_argument("--num_files_limit", type=int, default=1000,
                    help="Limit the number of files to process per batch. Default is 1000.")
parser.add_argument("--service_account_key_path", type=str, required=True,
                    help="Path to the GCP service account key file.")
parser.add_argument("--max_workers", type=int, default=multiprocessing.cpu_count(),
                    help="Maximum number of workers for concurrent processing.")
parser.add_argument("--folders", type=str, nargs='+', help="List of folders to iterate over.")
parser.add_argument("--max_files", type=int, default=None,
                    help="Maximum number of files to process in total. Default is None.")
parser.add_argument("--use_folder_limit", type=bool, default=True,
                    help="Whether to limit the number of files processed per folder.")

args = parser.parse_args()

assert args.model_name, "A model name must be provided."
assert args.hf_username, "A Hugging Face username must be provided."
assert args.hf_dataset_name, "A Hugging Face dataset name must be provided."

bucketName = args.bucket_name
pref = args.prefix
model_name = args.model_name
l_path = args.local_path
chunk_size = args.chunk_size_gb
hf_username = args.hf_username
hf_dataset_name = args.hf_dataset_name
checkpointFile = args.checkpoint_file
base_chunk_c = args.base_chunk_count
num_files_limit = args.num_files_limit
service_account = args.service_account_key_path
workers = args.max_workers
total_max_files = args.max_files
gsFolders = args.folders
folder_limit = args.use_folder_limit

# Settings.embed_model = embed_m

# lc_embed_model = HuggingFaceEmbeddings(
#     model_name=model_name
# )
# embed_m = LangchainEmbedding(lc_embed_model)


def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return set(json.load(f))
    return set()


def save_checkpoint(checkpoint_file, processed_folders):
    with open(checkpoint_file, 'w') as f:
        json.dump(list(processed_folders), f)


def list_files_in_bucket(bucket_name, folder, prefix, service_account_key_path):
    client = storage.Client.from_service_account_json(service_account_key_path)
    bucket = client.bucket(bucket_name)
    folder_prefix = f"{prefix}/{folder}"
    blobs = bucket.list_blobs(prefix=folder_prefix)
    for page in blobs.pages:
        yield from [blob.name for blob in page]


def process_files(file_paths, bucket_name, service_account_key_path, max_workers):
    global embed_m
    try:
        reader = GCSReader(
            bucket=bucket_name,
            service_account_key_path=service_account_key_path,
            key=file_paths,
        )

        documents = reader.load_data()

        print(f"Loaded {file_paths}")

        sentence_splitter = SentenceSplitter(
            chunk_size=256,
            chunk_overlap=20,
        )

        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=90,
            embed_model=embed_m,
            sentence_splitter=sentence_splitter.split_text,
            max_chunk_size=256,
            min_chunk_size=100,
            verbose=True
        )

        print(f"Extracting documents nodes from {file_paths}")

        nodes = splitter.get_nodes_from_documents(documents, show_progress=True)

        print(f"Extracted {len(nodes)} nodes from {file_paths}")

        print(f"Loading nodes chunks and metadata from {file_paths}")

        # chunks = [node.get_content() for node in nodes]
        chunks = [{
            "content": clean_text(node.get_content()),
            "file_name": node.metadata.get("file_name"),
            "node_id": node.id_
        } for node in nodes]

        print(f"Loaded {len(chunks)} chunks from {file_paths}")

        return chunks
    except Exception as e:
        print(f"Error processing {file_paths}: {e}")
        return []


def process_futures(futures, total_chunks, current_size, chunk_size_gb, chunk_count, temp_processed_files, processed_files, local_path, checkpoint_file, file_paths, lock):
    try:
        for future, file_path in zip(as_completed(futures), file_paths):
            file_chunks = future.result()
            file_size = sum(len(chunk["content"].encode('utf-8', errors='ignore')) for chunk in file_chunks) / (1024**3)  # Convert size to GB
            with lock:
                if current_size + file_size > chunk_size_gb or file_size > chunk_size_gb:
                    if current_size > 0:
                        print(f"Uploading chunk {chunk_count + 1}")
                        upload_dataset(total_chunks, chunk_count, local_path)
                        print(f"Uploaded chunk {chunk_count + 1}")
                        total_chunks = []
                        current_size = 0
                        chunk_count += 1
                        processed_files.update(temp_processed_files)
                        save_checkpoint(checkpoint_file, processed_files)
                        temp_processed_files = set()

                    if file_size > chunk_size_gb:
                        print(f"Uploading large file chunk {chunk_count + 1}")
                        upload_dataset(file_chunks, chunk_count, local_path)
                        print(f"Uploaded large file chunk {chunk_count + 1}")
                        chunk_count += 1
                        processed_files.add(file_path)
                        save_checkpoint(checkpoint_file, processed_files)
                        continue

                total_chunks.extend(file_chunks)
                current_size += file_size

    except Exception as e:
        print(f"Error occurred: {e}")
        if total_chunks:
            print(f"Uploading partial chunk {chunk_count + 1} due to error")
            upload_dataset(total_chunks, chunk_count, local_path)
            print(f"Uploaded partial chunk {chunk_count + 1}")
            processed_files.update(temp_processed_files)
            save_checkpoint(checkpoint_file, processed_files)


def process_main_directory(bucket_name, prefix, folders, chunk_size_gb, local_path, checkpoint_file, model_name, device, service_account_key_path, max_workers, max_files, use_folder_limit, lock):
    global embed_m
    embed_m = None  # Ensure the model is not initialized at the start

    processed_files = load_checkpoint(checkpoint_file)
    # folders = [os.path.join(main_directory, d) for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
    # folders.sort(reverse=True)
    total_chunks = []
    current_size = 0
    chunk_count = 0
    total_processed_files = 0
    # total_folders = len(folders)
    temp_processed_files = set()

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_model, initargs=(model_name, device)) as executor:
        futures = []
        file_paths = []
        for folder in folders:
            folder_files_processed = 0
            for file_path in list_files_in_bucket(bucket_name, folder, prefix, service_account_key_path):
                if file_path in processed_files:
                    print(f"Skipping already processed file {file_path}")
                    continue

                futures.append(executor.submit(process_files, file_path, bucket_name,
                                               service_account_key_path, max_workers))
                file_paths.append(file_path)
                temp_processed_files.add(file_path)
                total_processed_files += 1
                folder_files_processed += 1
                print(f"Total files processed: {total_processed_files}")

                if max_files is not None and total_processed_files >= max_files:
                    print(f"Reached the maximum number of files to process: {max_files}")
                    break

                if use_folder_limit and folder_files_processed >= num_files_limit:
                    print(f"Reached the limit of files to process per folder: {num_files_limit}")
                    break  # Move to the next folder

                if len(futures) >= num_files_limit:
                    process_futures(futures, total_chunks, current_size, chunk_size_gb, chunk_count, temp_processed_files, processed_files, local_path, checkpoint_file, file_paths, lock)
                    futures = []
                    file_paths = []

            # Handle case when max file limit is reached
            if max_files is not None and total_processed_files >= max_files:
                break

        # Process remaining futures after the loop
        if futures:
            process_futures(futures, total_chunks, current_size, chunk_size_gb, chunk_count, temp_processed_files, processed_files, local_path, checkpoint_file, file_paths, lock)

    if total_chunks:
        print(f"Uploading final chunk {chunk_count + 1}")
        upload_dataset(total_chunks, chunk_count, local_path)
        print(f"Uploaded final chunk {chunk_count + 1}")
        processed_files.update(temp_processed_files)
        save_checkpoint(checkpoint_file, processed_files)

    print(f"Total files processed: {total_processed_files}")
    print(f"Total batches uploaded: {chunk_count + 1}")


def upload_dataset(chunks, chunk_count, local_path):
    # data = {"text": chunks}
    data = {
        "text": [chunk["content"] for chunk in chunks],
        "file_name": [chunk["file_name"] for chunk in chunks],
        "node_id": [chunk["node_id"] for chunk in chunks]
    }

    try:
        dataset = Dataset.from_dict(data)
        dataset_dict = DatasetDict({"train": dataset})

        chunk_count += base_chunk_c

        local_path = f'{local_path}/dataset_chunk_{chunk_count}'
        dataset.save_to_disk(local_path)

        repo_name = f'{hf_username}/{hf_dataset_name}'
        hf_api = HfApi()

        if chunk_count == 0:
            hf_api.create_repo(repo_name, repo_type="dataset", exist_ok=True)

        dataset_dict.push_to_hub(repo_name, f'chunk_{chunk_count}', private=False, token=hfToken)

        # os.system(f"rm -rf {local_path}")
        shutil.rmtree(local_path)
        print(f"Deleted local path {local_path}")
    except UnicodeEncodeError as e:
        print(f"Failed to encode dataset chunk {chunk_count} due to encoding error: {e}")
        # Save problematic data locally for inspection
        with open(f'{local_path}/problematic_data_chunk_{chunk_count}.json', 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Failed to upload dataset chunk {chunk_count}: {e}")
        print(f"Local temp {local_path} not deleted to preserve data")


def init_model(model_name, device):
    global embed_m
    embed_m = HuggingFaceEmbedding(model_name=model_name, device=device, trust_remote_code=True)


def clean_text(text):
    # Remove or replace characters that can't be encoded in UTF-8
    return text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')


if __name__ == "__main__":
    lock = Lock()
    process_main_directory(bucket_name=bucketName, prefix=pref, folders=gsFolders,
                           chunk_size_gb=chunk_size, local_path=l_path,
                           checkpoint_file=checkpointFile, model_name=model_name,
                           service_account_key_path=service_account, max_workers=workers,
                           max_files=total_max_files, use_folder_limit=folder_limit, lock=lock, device='cuda')
