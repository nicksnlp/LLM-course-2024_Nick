# Fine-tuning T5 model with QA dataset

## References:
## https://chatgpt.com/share/6771e816-5bac-800b-ba78-dd785ad316b3


!pip install peft transformers datasets accelerate
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import wandb
from huggingface_hub import login, create_repo
from peft import SFTTrainer  # This would be the specialized trainer for supervised fine-tuning

# Step 1: Login to Hugging Face
login()

# Step 2: Login to Weights & Biases
wandb.login()

# Step 3: Create a Repository on Hugging Face (Optional)
repo_name = "t5-lora-hallucination"
create_repo(repo_name)

# Step 4: Load and Preprocess Dataset
dataset = load_dataset("qags")

def preprocess_function(example):
    context = example['context']
    summary = example['generated_text']
    hallucinated_spans = example['annotations']  # Hallucinated spans
    return {
        "model_input": context,
        "model_output_text": summary,
        "hallucinated_words": hallucinated_spans
    }

formatted_train_dataset = dataset['train'].map(preprocess_function)
formatted_val_dataset = dataset['validation'].map(preprocess_function)

# Step 5: Tokenize Dataset for T5
tokenizer = T5Tokenizer.from_pretrained("t5-base")

def tokenize_function(example):
    model_inputs = tokenizer(example["model_input"], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(example["model_output_text"], max_length=128, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized_train = formatted_train_dataset.map(tokenize_function, batched=True)
tokenized_val = formatted_val_dataset.map(tokenize_function, batched=True)

# Step 6: Load T5 Model and Apply LoRA
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],  # Apply LoRA to query and value modules
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Step 7: Define Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./t5-lora-hallucination",  # Local directory
    evaluation_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    predict_with_generate=True,
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=100,
    report_to="wandb",  # Enable logging to Weights & Biases
    logging_dir="./logs",  # Directory to store the logs
    push_to_hub=True,  # Push the model to Hugging Face Model Hub
    hub_model_id=repo_name  # Hugging Face model repository name
)

# Step 8: Use SFTTrainer for Supervised Fine-Tuning
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# Step 9: Save the Fine-Tuned Model and Tokenizer to Hugging Face
model.save_pretrained(f"./{repo_name}")
tokenizer.save_pretrained(f"./{repo_name}")

# Push model and tokenizer to Hugging Face
api = HfApi()
api.upload_folder(
    repo_id=repo_name,  # Model name on Hugging Face Hub
    folder_path=f"./{repo_name}",  # Path to local model folder
    path_in_repo=""  # Upload all files in the folder
)

print(f"Model and tokenizer have been uploaded to Hugging Face Hub: {repo_name}")
