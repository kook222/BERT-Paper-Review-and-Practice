# -*- coding: utf-8 -*-
# [체험] 내 맘대로 문장 넣어서 테스트하는 코드 (BERT-Large 버전)
# 파일명: 3_predict_bert_large.py
import torch
import os
import glob
from transformers import BertTokenizer, BertForSequenceClassification

# 1. 학습된 BERT-Large 모델 경로 찾기
base_path = "./bert_large_result"
model_path = base_path

# 만약 폴더 안에 모델 파일이 바로 없으면, 가장 최신 체크포인트(checkpoint)를 자동으로 찾습니다.
if not os.path.exists(os.path.join(base_path, "config.json")) and \
   not os.path.exists(os.path.join(base_path, "model.safetensors")):
    
    checkpoints = glob.glob(f"{base_path}/checkpoint-*")
    if checkpoints:
        model_path = max(checkpoints, key=os.path.getmtime) # 가장 최근 저장된 폴더 선택

print(f"? BERT-Large 모델 로딩 중... ({model_path})")
print("   (모델이 커서 로딩에 시간이 조금 걸릴 수 있습니다)")

try:
    # BERT-Large 전용 토크나이저
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    model = BertForSequenceClassification.from_pretrained(model_path)
except OSError:
    print("? 학습된 모델이 없습니다. 1_train_bert_large.py를 먼저 실행하세요.")
    exit()

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)
model.eval() # 평가 모드

def predict_sentiment(text):
    # 입력 처리
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    score = probs[0].tolist() # [부정확률, 긍정확률]
    
    return score

# 2. 사용자 입력 루프
print("\n? 영어 문장을 입력하세요 (종료하려면 'quit' 입력)")
print("-" * 60)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input.strip():
            continue

        scores = predict_sentiment(user_input)
        neg_score = scores[0] * 100
        pos_score = scores[1] * 100

        # BERT-Large라고 명시
        print(f"BERT-Large: ? 부정 {neg_score:.1f}%  |  ? 긍정 {pos_score:.1f}%")
        
        if pos_score > neg_score:
            print("            ? 결과: 긍정 (Positive) ?")
        else:
            print("            ? 결과: 부정 (Negative) ?")
        print("-" * 60)
        
    except KeyboardInterrupt:
        print("\n? 종료합니다.")
        break
    except Exception as e:
        print(f"?? 에러 발생: {e}")