"""
Author: Amr Sherif
Version: 1.0.0
Date: 2024-06-13
Description: This script provides utility functions for generating synthetic instruction-following datasets
             based on arXiv full-text chunks. It includes templates for different instruction types and
             functions for interacting with the Together AI API.

Project: Arxiv Assistant
Dependencies: random, logging
License: MIT License

Change Log:
    - Version 1.0.0: Initial version with instruction templates, API interaction functions, and chunk processing logic.
"""

import random
import logging

logging.basicConfig(level=logging.INFO)

def summarization_template(chunk):
    instruction = "Summarize the following text:"
    return {
        "instruction": instruction,
        "input": chunk,
        "output": "summary of the chunk"
    }

def question_answering_template(chunk):
    instruction = "Answer the following question based on the text:"
    question = "What is the main point discussed? be clear, concise, and structured."
    return {
        "instruction": instruction + " " + question,
        "input": chunk,
        "output": "answer to the question"
    }

def information_extraction_template(chunk):
    instruction = "Extract the main key points and takeaways from the following text:"
    return {
        "instruction": instruction,
        "input": chunk,
        "output": "information extracted from the chunk"
    }

templates = [summarization_template, question_answering_template, information_extraction_template]

def generate_output_completion(prompt, client):
    response = client.completions.create(
        model="mistralai/Mixtral-8x22B-Instruct-v0.1",
        prompt=prompt,
        max_tokens=1000
    )
    return response.choices[0].text.strip()

def generate_output_chat(prompt, client):
    response = client.chat.completions.create(
        model="mistralai/Mixtral-8x22B-Instruct-v0.1",
        messages=[prompt],
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

def process_chunk(item, client):
    chunk = item['text']
    template = random.choice(templates)
    datapoint = template(chunk)
    datapoint["output"] = generate_output_chat(
        {"role": "user", "content": datapoint["instruction"] + " " + datapoint["input"]}, client)
    return datapoint
