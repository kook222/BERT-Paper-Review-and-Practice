# -*- coding: utf-8 -*-
# [체험] 내 맘대로 문장 넣어서 테스트하는 코드
# 파일명: 3_predict.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 1. 학습된 내 모델 불러오기
model_path = "./bert_final_model" # 학습 완료 후 생성된 폴더
print(f"? 모델 로딩 중... ({model_path})")

try:
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
except OSError:
    print("? 학습된 모델이 없습니다. 1_train.py를 먼저 실행하세요.")
    exit()

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    score = probs[0].tolist() # [부정확률, 긍정확률]
    
    return score

# 2. 사용자 입력 루프
print("\n? 영어 문장을 입력하세요 (종료하려면 'quit' 입력)")
print("-" * 50)

while True:
    user_input = input("User: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    
    if not user_input.strip():
        continue

    scores = predict_sentiment(user_input)
    neg_score = scores[0] * 100
    pos_score = scores[1] * 100

    print(f"BERT: ? 부정 {neg_score:.1f}%  |  ? 긍정 {pos_score:.1f}%")
    if pos_score > neg_score:
        print("      ? 결과: 긍정 (Positive) ?")
    else:
        print("      ? 결과: 부정 (Negative) ?")
    print("-" * 50)