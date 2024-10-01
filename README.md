# arXiv Assistant: Mistral-7B Fine-tuned for Scientific Paper Retrieval & Analysis

## Project Overview

This project showcases the development of an advanced language model based on Mistral-7B, fine-tuned specifically for analyzing and interacting with scientific papers from the arXiv repository.
It serves as a comprehensive demonstration of how to build and fine-tune a powerful yet efficient model for a domain-specific downstream task, leveraging both domain-specific data and synthetically generated datasets.  

### Development Process and Techniques

The development of this model involved several key steps and techniques:

1. **Dataset Preparation**: We processed a large corpus of arXiv papers, converting PDFs to text and chunking them into manageable sizes. This data was then uploaded to Hugging Face Datasets for efficient access during training.

2. **Synthetic Data Generation**: To enhance the model's instruction-following capabilities, we created a synthetic dataset using the processed arXiv data. This dataset includes various tasks such as summarization, question-answering, and information extraction specific to scientific papers.

3. **Fine-tuning**: We employed the LoRA (Low-Rank Adaptation) technique to fine-tune the Mistral-7B-Instruct-v0.2 model. This method allows for efficient adaptation of the large language model to our specific domain while minimizing the number of trainable parameters.

4. **Evaluation**: The model was evaluated using metrics such as BLEU, ROUGE, and perplexity, with a focus on its performance on scientific text understanding and generation tasks.

5. **Deployment**: We implemented deployment strategies using vLLM for high-performance inference, allowing the model to be easily integrated into various applications.

### arXiv Assistant: A RAG Assistant Prototype

This fine-tuned model serves as the underlying language model in the [arXiv Assistant: A RAG Assistant Prototype](https://github.com/amr-sheriff/arxiv-assistant).  

This Assistant is a demonstration project that showcases how to build a Retrieval-Augmented Generation (RAG) Assistant using a combination of open-source technologies, frameworks, and this fine-tuned Large Language Model.
While focused on arXiv papers, this approach can be adapted to various business-specific use cases, illustrating its versatility.

The arXiv Assistant can be used in various ways, including:

1. Querying scientific papers
2. Summarizing research articles
3. Extracting key information from papers
4. Answering questions about specific research topics
5. Assisting in literature reviews and research analysis

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Baseline Model](#baseline-model)
5. [Dataset Preparation](#dataset-preparation)
6. [Synthetic Data Generation](#synthetic-data-generation)
7. [Model Fine-tuning](#model-fine-tuning)
8. [Evaluation](#evaluation)
9. [Deployment](#deployment)
10. [Contributing](#contributing)
11. [License](#license)

## Project Structure

```
arxiv-assistant-mistral-7b-finetuned/
├── baseline/
│   ├── arxiv_assistant_baseline.ipynb
│   ├── helpers.py
├── datasets/
│   ├── arxiv_dataset.ipynb
│   ├── process_arxiv_data.py
│   ├── checkpoint.json
│   ├── temp/
├── finetuning/
│   ├── lora_finetuning.py
│   ├── install-packages.sh
├── instruct_synthetic_dataset/
│   ├── arxiv_instruct_synthetic_dataset.ipynb
│   ├── utils.py
│   ├── synthetic_finetuning_data.jsonl
├── eval/
│   ├── model_eval_batch_inference.ipynb
│   ├── test-data-predictions-ref.jsonl
│   ├── eval_results.jsonl
│   ├── models/
├── deployment_endpoints/
│   ├── vllm_inference_server_ep.ipynb
│   ├── text_embedding_inference_ep.ipynb
├── .env
├── .env.example
├── README.md
├── requirements.txt
├── LICENSE
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/amr-sheriff/arxiv-assistant-mistral-7b-finetuned.git
   cd arxiv-assistant-mistral-7b-finetuned
   ```

2. Set up a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Copy the `.env.example` file to `.env` and fill in the necessary API keys and paths.

## Baseline Model

The baseline model serves as a starting point for the project. It provides a comprehensive qualitative assessment of the baseline performance of vanilla `Mistral-7B-v0.1` and `Mistral-7B-Instruct-v0.2` for the downstream task of scientific paper retrieval and analysis.

To use the baseline model:

1. Navigate to the `baseline/` directory.
2. Open the `arxiv_assistant_baseline.ipynb` notebook.
3. Follow the instructions in the notebook to load and use the baseline Mistral-7B model.

The `helpers.py` file contains utility functions used in the baseline notebook.

## Dataset Preparation

The dataset preparation process involves fetching arXiv papers from Google Cloud Storage, processing the PDFs into text chunks, and creating a dataset on Hugging Face Datasets. The dataset is available [here](https://huggingface.co/datasets/amrachraf/arXiv-full-text-chunked).

To prepare the dataset:

1. Navigate to the `datasets/` directory.
2. Open the `arxiv_dataset.ipynb` notebook.
3. Set up Google Cloud credentials and configure the bucket name and prefix.
4. Follow the instructions in the notebook and run the notebook cells to list and explore files in the bucket.
5. Execute the `process_arxiv_data.py` script with appropriate arguments:

```bash
!python process_arxiv_data.py --bucket_name $bucket_name --prefix $prefix --folders 2403 2402 2401 2312 2311 2310 2309 2308 2307 \
2306 2305 2304 2303 2302 2301 2212 2211 2210 2209  \
 --service_account_key_path $service_account_key_path \
 --model_name "WhereIsAI/UAE-Large-V1" --hf_username "your_username" \
 --hf_dataset_name "arXiv-full-text-chunked" --chunk_size_gb 0.1 \
 --local_path "TEMP_DIR_PATH" \
 --base_chunk_count 4 --num_files_limit 200 --max_files 400 --use_folder_limit True
```

This script will process the arXiv PDFs, chunk the text, and upload the dataset to Hugging Face Datasets.

## Synthetic Data Generation

To generate the synthetic [instruction tuning dataset](https://huggingface.co/datasets/amrachraf/arXiv-full-text-synthetic-instruct-tune): 

1. Navigate to the `instruct_synthetic_dataset/` directory.
2. Open the `arxiv_instruct_synthetic_dataset.ipynb` notebook.
3. Run the cells to load the arXiv dataset and generate synthetic instruction-following data.
4. The synthetic data will be saved in `synthetic_finetuning_data.jsonl`.

The `utils.py` file contains helper functions for data processing and prompt generation.

## Model Fine-tuning

The model is fine-tuned using LoRA (Low-Rank Adaptation) technique. The model is available on Hugging Face [here](https://huggingface.co/amrachraf/arxiv-assistant-merged_peft_model).

To fine-tune the model:

1. Navigate to the `finetuning/` directory.
2. Ensure all required packages are installed by running:
   ```
   bash install-packages.sh
   ```
3. Open and run the `lora_finetuning.py` script:
   ```
   python lora_finetuning.py
   ```

The fine-tuning process uses the Mistral-7B-Instruct-v0.2 model as a base and applies LoRA to adapt it for the arXiv domain. The script includes the following key components:

- Data loading and preprocessing
- Model and tokenizer initialization
- LoRA configuration
- Training arguments setup
- Custom evaluation metrics (BLEU, ROUGE, perplexity)
- Integration with Weights & Biases for experiment tracking

## Evaluation

Model evaluation is performed using various metrics, including BLEU, ROUGE, and perplexity. To evaluate the model:

1. Navigate to the `eval/` directory.
2. Open the `model_eval_batch_inference.ipynb` notebook.
3. Run the cells to load the fine-tuned model and perform batch inference on the test dataset.
4. The notebook uses vLLM for efficient batch inference.
5. Evaluation results are computed and stored in `eval_results.jsonl`.
6. The notebook also generates `test-data-predictions-ref.jsonl`, which contains the model's predictions alongside reference outputs.

Key components of the evaluation process:
- Loading the merged PEFT model
- Setting up vLLM for batch inference
- Computing BLEU, ROUGE, and perplexity scores
- Analyzing and visualizing the results

## Deployment

The project includes deployment endpoints for inference:

1. VLLM Inference Server:
    - Navigate to the `deployment_endpoints/` directory.
    - Open `vllm_inference_server_ep.ipynb`.
    - This notebook demonstrates how to set up and use a vLLM inference server for high-performance serving of the fine-tuned model.
    - It includes examples of using both the OpenAI client and direct HTTP requests for inference.

2. Text Embedding Inference: (WIP)
    - In the same directory, open `text_embedding_inference_ep.ipynb`.
    - This notebook shows how to generate text embeddings using an embeddings model, which will be used in the RAG process and various downstream tasks.

Both endpoints can be adapted for production deployment on cloud platforms or local servers.

## Contributing

Contributions to the arXiv Assistant project are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear, descriptive messages.
4. Push your changes to your fork.
5. Submit a pull request to the main repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For any questions or issues, please open an issue on the GitHub repository. We appreciate your contributions and feedback!