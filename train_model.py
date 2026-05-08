import pandas as pd
import re
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import torch

MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "data/WELFake_Dataset.csv"
OUTPUT_DIR = "model"

print("📊 Loading dataset...")
df = pd.read_csv(DATA_PATH)
df = df.dropna().drop_duplicates()
df["Content"] = (df["title"].astype(str) + " " + df["text"].astype(str)).str.lower()
df["Content"] = df["Content"].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))

dataset = Dataset.from_pandas(df[["Content", "label"]])
dataset = dataset.rename_column("label", "labels")

split = dataset.train_test_split(test_size=0.2, seed=42)
val_test = split["test"].train_test_split(test_size=0.5, seed=42)

final_ds = DatasetDict({
    "train": split["train"],
    "validation": val_test["train"],
    "test": val_test["test"]
})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    return tokenizer(examples["Content"], truncation=True, padding=True, max_length=512)

print("🔤 Tokenizing...")
tokenized = final_ds.map(tokenize, batched=True, remove_columns=["Content"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
    report_to="none"
)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    return {"accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"]}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("🚀 Training started... (This may take 30-90 minutes)")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Model successfully trained and saved in 'model/' folder!")