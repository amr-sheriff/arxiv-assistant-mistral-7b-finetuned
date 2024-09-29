# -*- coding: utf-8 -*-

from huggingface_hub import notebook_login, login
# from google.colab import userdata
from datasets import load_dataset
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from sklearn.metrics import (accuracy_score,
                            confusion_matrix,
                            classification_report)
from dotenv import load_dotenv

load_dotenv()

hfToken = os.getenv('hf')
wandbToken = os.getenv('wandb')

login(hfToken, add_to_git_credential=True, new_session=False)

# A wrapper function which will get completion by the model from a user's query
def get_completion(query: str, instuction: str,
                   model, tokenizer,
                   add_special_tokens: bool=True,
                   return_tensors: str="pt", max_new_tokens: int=1000,
                   do_sample: bool=True, device: str="cuda:0") -> str:

  device = device

  prompt_template = """
  <s>[INST] {instuction}
  {query} [/INST]
  """
  prompt = prompt_template.format(query=query, instuction=instuction)

  encodeds = tokenizer(prompt,
                       return_tensors=return_tensors,
                       add_special_tokens=add_special_tokens
                       )

  model_inputs = encodeds.to(device)


  generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tokenizer.eos_token_id)
  decoded = tokenizer.batch_decode(generated_ids)
  return (decoded[0])

device = "cuda:0"

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
    # bnb_4bit_compute_dtype=torch.bfloat16
)

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config=bnb_config,
                                             device_map=device,
                                            #  device_map={"":0},
                                             trust_remote_code=True,
                                             token=hfToken
                                             )

tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          use_fast=True,
                                          token=hfToken,
                                          trust_remote_code=True,
                                          add_eos_token=True
                                          )

# tokenizer.padding_side = 'left'
tokenizer.padding_side = 'right'
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = tokenizer.unk_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

# tokenizer.pad_token_id=2041
model.config.pad_token_id = tokenizer.pad_token_id

instuction = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
query = "Can I find information about the code's approach to handling long-running tasks and background jobs?"
# response = "Yes, the code includes methods for submitting jobs, checking job status, and retrieving job results. It also includes a method for canceling jobs. Additionally, there is a method for sampling multiple outputs from a model, which could be useful for long-running tasks."
# query2 = 'Explain in detail.'

result = get_completion(query=query, instuction=instuction, model=model, tokenizer=tokenizer, add_special_tokens=False)
print(result)

data = load_dataset("amrachraf/arXiv-full-text-synthetic-instruct-tune", split="train")

# small dataset for testing
test = True
if test:
    data = data.select(range(100))

# Explore the data
df = data.to_pandas()
df.head(10)

system_instruction = """You are an arXiv assistant, your name is Marvin. You provide detailed, comprehensive and helpful responses to any request,
specially requests related to scientific papers published on arXiv, structure your responses and reply in a clear scientific manner.
Ensure to greet the user at the start of the first message of the conversation only. And ensure to ask the user if your response was clear and sufficient and if he needs any other help.
As an arXiv assistant, Your task is to generate an appropriate response based on the conversation and context given.
The tone of your answer should be warm, kind and friendly."""

def generate_prompt(example, system_instruction):
    return f"""<s>[INST]<<SYS>> {system_instruction} <</SYS>>
    {example['instruction']}
    {example['input']} [/INST]
    {example['output']} </s>""".strip()

def generate_test_prompt(example, system_instruction):
    return f"""<s>[INST]<<SYS>> {system_instruction} <</SYS>>
    {example['instruction']}
    {example['input']} [/INST]""".strip()

data = data.train_test_split(test_size=0.05, seed=3, shuffle=True)
train_data = data["train"]
test_data = data["test"]

train_data.shape, test_data.shape

prompt_column_train = [generate_prompt(example, system_instruction) for example in train_data]
train_data = train_data.add_column("prompt", prompt_column_train)
# train_data = train_data.add_column("text", prompt_column_train)
train_data = train_data.map(lambda samples: tokenizer(samples["prompt"]), batched=True)

prompt_column_test = [generate_test_prompt(example, system_instruction) for example in test_data]
test_data = test_data.add_column("prompt", prompt_column_test)
# test_data = test_data.add_column("text", prompt_column_test)
test_data = test_data.map(lambda samples: tokenizer(samples["prompt"]), batched=True)

test_data, train_data

print(train_data[0]['prompt'])

import datetime as dt

datet = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
project_name = 'arxiv-assistant-instruct-tune'
base_model_name = "mistral-7b-instruct-v0.2"
run_name = base_model_name + "-" + project_name + "-" + datet
output_dir = "./" + run_name

model.config.use_cache = False # silence warnings, re-enable for inference
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

print(model)

import bitsandbytes as bnb
def find_all_linear_names(model):
  cls = bnb.nn.Linear4bit # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
  lora_module_names = set()
  for name, module in model.named_modules():
    if isinstance(module, cls):
      names = name.split('.')
      lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
      lora_module_names.remove('lm_head')
  return list(lora_module_names)

modules = find_all_linear_names(model)
print(modules)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64,
    lora_alpha=8,
    target_modules=modules,
    lora_dropout=0.03,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

print(model)

import transformers
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

training_arguments = transformers.Seq2SeqTrainingArguments(
# training_arguments = SFTConfig(
    predict_with_generate=True, # only w\ seq2seq trainer
    generation_max_length=1000, # only w\ seq2seq trainer
    generation_config=transformers.GenerationConfig(max_new_tokens=1000),
    output_dir=output_dir,
    logging_dir = "logs",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    eval_accumulation_steps=40,
    num_train_epochs=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_strategy='steps',
    save_steps=1000,
    # save_total_limit=5,
    logging_strategy='steps',
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio = 0.05,
    group_by_length=True, # only w\ SFTConfig
    lr_scheduler_type="cosine",
    report_to="wandb",
    eval_strategy="steps",
    eval_steps=1000,
    # eval_strategy="epoch",
    do_eval=True,
    run_name = run_name,
    push_to_hub = True,
    hub_model_id = "arxiv-assistant-mistral7b",
    hub_token=hfToken,
    hub_strategy="checkpoint",
    disable_tqdm=False,
    auto_find_batch_size=True,
    eval_do_concat_batches=True,
    fp16_full_eval=True,
    gradient_checkpointing=True,
    dataloader_num_workers=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    include_inputs_for_metrics=True,
)

import evaluate

metrics = evaluate.combine(
    ["bleu", "rouge", "perplexity"], force_prefix=True
)

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    predictions = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    references = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    outputs = metrics.compute(predictions=predictions,
                             references=references)

    # Log metrics to wandb
    wandb.log(outputs)

    return outputs

# trainer = transformers.Trainer(
#     model=model,
#     train_dataset=train_data,
#     eval_dataset=test_data,
#     args=transformers.TrainingArguments(
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=4,
#         warmup_steps=0.03,
#         max_steps=100,
#         learning_rate=2e-4,
#         fp16=True,
#         logging_steps=1,
#         output_dir=output_dir,
#         optim="paged_adamw_8bit",
#         save_strategy="epoch",
#     ),
#     data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
# )

trainer = transformers.Seq2SeqTrainer(
# trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    # peft_config=lora_config, # only w\ SFTTrainer
    # dataset_text_field="prompt", # only w\ SFTTrainer
    # tokenizer=tokenizer,
    data_collator=DataCollatorForCompletionOnlyLM(instruction_template='[INST]', response_template='[/INST]', tokenizer=tokenizer, mlm=False), # only w\ seq2seq trainer
    # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    args=training_arguments,
    # callbacks=callbacks,
    # packing=False, # only w\ SFTTrainer
    compute_metrics=compute_metrics
    # eval_packing=False
    # max_seq_length=512
    )

train_arg_dict = training_arguments.to_dict()
train_arg_dict

lora_config_dict = lora_config.to_dict()
lora_config_dict

wandb_config_tracker = {
    "train_arg_dict": train_arg_dict,
    "lora_config_dict": lora_config_dict
}

import wandb
wandb.login(key = wandbToken)
run = wandb.init(
    project=project_name,
    job_type="training",
    anonymous="allow",
    config=wandb_config_tracker,
    name=run_name,
)

trainer.train()

trainer.push_to_hub()
wandb.finish()
model.config.use_cache = True

model.eval()