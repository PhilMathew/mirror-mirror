from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from pathlib import Path

def train_qa_model(train_dataset:Dataset, 
                   model_name:str, 
                   prompt_template:str, 
                   output_dir : Path, 
                   num_train_epochs: int,
                   per_device_train_batch_size: int,
                   max_length :int = 512,
                ):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # --- Format + Tokenize ---
    def format_and_tokenize(example):
        text = prompt_template.format(question=example["question"], answer=example["answer"])
        return tokenizer(text, truncation=True, padding="max_length", max_length=max_length)

    tokenized_train = train_dataset.map(format_and_tokenize, remove_columns=train_dataset.column_names)

    # --- Data Collator ---
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Training ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=f'training_{model_name}',
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        logging_steps=15,
        save_strategy="epoch",
        save_total_limit=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir / "final_model")