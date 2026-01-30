# -*- coding: utf-8 -*-
# 파일명: 3_predict_gpt.py
import torch
from transformers import OpenAIGPTTokenizer, OpenAIGPTForSequenceClassification

# 1. 설정 (학습된 GPT 모델 폴더 경로)
model_path = "./gpt_result"  # GPT 학습이 끝나면 이 폴더가 생깁니다.
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

print(f"🔥 사용 장치: {device}")
print("📂 학습된 GPT 모델을 불러오는 중...")

try:
    # 2. 모델과 토크나이저 불러오기
    # [주의] GPT는 패딩 토큰이 없어서 학습 때처럼 unk_token으로 설정해줘야 함
    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    tokenizer.pad_token = tokenizer.unk_token
    
    # 저장된 체크포인트(checkpoint) 폴더가 아니라, trainer.save_model()로 저장한 최종 모델을 불러와야 함
    # 만약 에러가 나면 model_path를 "./gpt_result/checkpoint-xxxx" 형태로 바꿔보세요.
    model = OpenAIGPTForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval() # 평가 모드로 전환

except Exception as e:
    print(f"❌ 모델을 불러올 수 없습니다. 학습이 먼저 완료되어야 합니다.\n에러: {e}")
    exit()

print("✅ GPT 모델 로딩 완료! (종료하려면 'q' 입력)")

# 3. 예측 루프
labels = {0: "👎 부정 (Negative)", 1: "👍 긍정 (Positive)"}

while True:
    text = input("\n📝 영화 리뷰를 영어로 입력하세요: ")
    if text.lower() == 'q':
        break
    
    # 입력 데이터 전처리
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 예측
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred].item() * 100

    print(f"   🤖 GPT의 판단: {labels[pred]} (확신: {confidence:.2f}%)")