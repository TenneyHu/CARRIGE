from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def tokenize_function(batch):
    texts = [
        "Nombre: " + nombre + ". Ingredientes: " + ingredientes
        for nombre, ingredientes in zip(batch["Nombre"], batch["Ingredientes"])
    ]
    return tokenizer(texts, truncation=True, padding="max_length")

def label_by_country(example):
    return {"label": 1 if example["Pais"] == "ESP" else 0}

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def compute_metrics_eval(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

model_name = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
#for train
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
#for test
model = BertForSequenceClassification.from_pretrained("./CAS_model/checkpoint-276")

ds = load_dataset("somosnlp/RecetasDeLaAbuela", "version_1")
ds = ds.filter(lambda example: example["Pais"] is not None and example["Pais"].strip() != "")
ds = ds.map(label_by_country)
tokenized_dataset = ds.map(tokenize_function, batched=True)

split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

training_args = TrainingArguments(
    output_dir="./CAS_model",
    logging_strategy="no",
    evaluation_strategy="epoch",           
    save_strategy="epoch",                 
    num_train_epochs=3,                    
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=2e-5,
    weight_decay=0.01,
    metric_for_best_model="f1",           
    save_total_limit=1,                    
    report_to="none"                       
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics_eval,
)

#trainer.train()
trainer.evaluate(eval_dataset=test_dataset)
eval_results = trainer.evaluate(eval_dataset=test_dataset)
for k, v in eval_results.items():
    print(f"{k}: {v:.4f}")