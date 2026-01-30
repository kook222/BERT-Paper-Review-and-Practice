# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import os
import json
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate

# 1. 설정
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"🔥 학습 장치: {device} (모델: BERT-Large - 끝판왕)")

# 2. 데이터 준비 (기존 파일 재사용)
print("📥 데이터 로딩 중...")
if not os.path.exists("train.parquet"):
    print("   -> (혹시 몰라 다운로드 코드 유지) 학습 데이터 받는 중...")
    df_train = pd.read_parquet("https://huggingface.co/datasets/imdb/resolve/main/plain_text/train-00000-of-00001.parquet")
    df_train.to_parquet("train.parquet")
    df_test = pd.read_parquet("https://huggingface.co/datasets/imdb/resolve/main/plain_text/test-00000-of-00001.parquet")
    df_test.to_parquet("test.parquet")

data_files = {"train": "train.parquet", "test": "test.parquet"}
dataset = load_dataset("parquet", data_files=data_files)

train_dataset = dataset["train"].shuffle(seed=42)
eval_dataset = dataset["test"].shuffle(seed=42).select(range(1000))

# 3. 모델 설정 (BERT-Large)
model_name = "bert-large-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("⚙️ 데이터 전처리 중...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# 4. 평가 함수
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 5. 학습 설정
training_args = TrainingArguments(
    output_dir="./bert_large_result",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    # [중요] 모델이 커서 메모리 터질까봐 16 -> 8로 줄였습니다.
    # 만약 그래도 "Out of Memory" 에러가 나면 4로 더 줄이세요.
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs_bert_large',
    dataloader_pin_memory=False 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

# 6. 학습 실행
print("\n🚀 [BERT-Large] 학습 시작! (시간이 Base보다 2~3배 더 걸립니다)")
trainer.train()

# 7. 기록 저장
history = []
for log in trainer.state.log_history:
    if 'eval_accuracy' in log:
        history.append({'epoch': log['epoch'], 'accuracy': log['eval_accuracy']})

with open('training_history_bert_large.json', 'w') as f:
    json.dump(history, f)

print("\n✅ BERT-Large 학습 완료!")