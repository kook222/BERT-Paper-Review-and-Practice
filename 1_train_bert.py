# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import os
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate

# 1. 맥북 가속(MPS) 확인
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"🔥 학습 장치: {device}")

# 2. 데이터셋 준비 (네트워크 우회: 로컬 저장 후 로딩)
print("📥 데이터 다운로드 중... (로컬 파일 사용)")

try:
    if not os.path.exists("train.parquet"):
        print("   -> 학습 데이터(train) 다운로드 중...")
        df_train = pd.read_parquet("https://huggingface.co/datasets/imdb/resolve/main/plain_text/train-00000-of-00001.parquet")
        df_train.to_parquet("train.parquet")
    
    if not os.path.exists("test.parquet"):
        print("   -> 평가 데이터(test) 다운로드 중...")
        df_test = pd.read_parquet("https://huggingface.co/datasets/imdb/resolve/main/plain_text/test-00000-of-00001.parquet")
        df_test.to_parquet("test.parquet")

    data_files = {"train": "train.parquet", "test": "test.parquet"}
    dataset = load_dataset("parquet", data_files=data_files)

except Exception as e:
    print(f"❌ 데이터 준비 중 오류 발생: {e}")
    exit()

# 학습은 전체 다 (25,000개) 사용
train_dataset = dataset["train"].shuffle(seed=42) 
# 평가는 1,000개만 사용 (속도 향상)
eval_dataset = dataset["test"].shuffle(seed=42).select(range(1000))

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("⚙️ 데이터 전처리 중...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# 3. 모델 불러오기
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# 4. 정확도 계산 함수
metric = evaluate.load("accuracy") 

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 5. 학습 설정
training_args = TrainingArguments(
    output_dir="./bert_result",
    eval_strategy="epoch",  # 최신 버전 호환 (evaluation_strategy -> eval_strategy)
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,          
    weight_decay=0.01,
    # [삭제함] use_mps_device=True (이 옵션은 이제 필요 없고 에러를 유발해서 삭제했습니다)
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

# 6. [Before] 학습 전 상태 평가
print("\n🧐 [Before] 학습 전 모델 성능 측정 중...")
init_metrics = trainer.evaluate()
print(f"   -> 학습 전 정확도: {init_metrics['eval_accuracy']:.4f}")

# 7. [Training] 학습 시작
print("\n🚀 [Start] 학습 시작! (중단된 곳부터 이어하기)")
# resume_from_checkpoint=True 옵션을 넣으면, bert_result 폴더를 뒤져서 가장 최신 파일부터 시작합니다.
train_result = trainer.train(resume_from_checkpoint=True)

# 8. [History] 그래프를 위해 기록 저장
history = []
history.append({'epoch': 0, 'accuracy': init_metrics['eval_accuracy']})

for log in trainer.state.log_history:
    if 'eval_accuracy' in log:
        history.append({'epoch': log['epoch'], 'accuracy': log['eval_accuracy']})

import json
with open('training_history.json', 'w') as f:
    json.dump(history, f)

print("\n✅ 학습 완료! 모델과 기록이 저장되었습니다.")
trainer.save_model("./bert_final_model")