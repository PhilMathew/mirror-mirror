import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from tqdm import tqdm
import numpy as np
import evaluate
from pathlib import Path
from mirrormirror.components.qa_datasets import load_tofu_entity_split
from mirrormirror.utils.train_llm_utils import train_qa_model
import os
import json 

def main():
    # --- Config ---
    model_name = "microsoft/phi-2"
    # model_name = "FacebookAI/roberta-base"
    prompt_template = "### Question:\n{question}\n### Answer:\n{answer}\n"
    dataset = "locuslab/tofu"
    author_idx = 4
    full_dataset = load_dataset(dataset, "full")
    output_dir_full = Path('/fendlnm1/mirror-mirror/results/tofu_forgetting/full_dataset_model')
    train_qa_model(full_dataset['train'], model_name, prompt_template, output_dir_full, num_train_epochs=3, per_device_train_batch_size=4)
    dataset_split = load_tofu_entity_split(author_idx)
    output_dir_split = Path(f'results/tofu_forgetting/author_{author_idx}')
    train_qa_model(dataset_split, model_name, prompt_template, output_dir_split, num_train_epochs=3, per_device_train_batch_size=4)


        # subset = "retain99"
       



if __name__ == '__main__':
    main()