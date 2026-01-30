# -*- coding: utf-8 -*-
# 파일명: final_graph.py
import json
import matplotlib
matplotlib.use('TkAgg') # 맥북 창 띄우기용
import matplotlib.pyplot as plt

def load_history(filename):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        epochs = [entry['epoch'] for entry in data]
        accs = [entry['accuracy'] for entry in data]
        return epochs, accs
    except FileNotFoundError:
        print(f"⚠️ {filename} 파일이 없습니다. (해당 모델 학습 안 됨)")
        return [], []

# 1. 데이터 3개 다 불러오기
print("📊 3개 모델 데이터 로딩 중...")
gpt_ep, gpt_acc = load_history('training_history_gpt.json')         # 1. OpenAI GPT
base_ep, base_acc = load_history('training_history.json')           # 2. BERT-Base
large_ep, large_acc = load_history('training_history_bert_large.json') # 3. BERT-Large

# 2. 그래프 그리기
plt.figure(figsize=(12, 7))

# (1) OpenAI GPT (주황 점선)
if gpt_ep:
    plt.plot(gpt_ep, gpt_acc, marker='s', linestyle='--', color='orange', label='OpenAI GPT', linewidth=2)
    # 마지막 값 표시
    plt.annotate(f"{gpt_acc[-1]:.1%}", (gpt_ep[-1], gpt_acc[-1]), textcoords="offset points", xytext=(10,0), color='orange', fontweight='bold')

# (2) BERT-Base (파란 실선)
if base_ep:
    plt.plot(base_ep, base_acc, marker='o', linestyle='-', color='blue', label='BERT-Base', linewidth=2)
    plt.annotate(f"{base_acc[-1]:.1%}", (base_ep[-1], base_acc[-1]), textcoords="offset points", xytext=(10,0), color='blue', fontweight='bold')

# (3) BERT-Large (빨간 굵은 실선 - 주인공!)
if large_ep:
    plt.plot(large_ep, large_acc, marker='*', linestyle='-', color='red', label='BERT-Large', linewidth=3, markersize=10)
    plt.annotate(f"{large_acc[-1]:.1%}", (large_ep[-1], large_acc[-1]), textcoords="offset points", xytext=(10,0), color='red', fontweight='bold')

# 3. 디자인
plt.title('Final Showdown: GPT vs BERT-Base vs BERT-Large (IMDb)', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.ylim(0.80, 0.95) # 80% ~ 95% 구간 집중 확대

# 4. 저장 및 출력
plt.savefig('final_comparison.png')
print("✅ 최종 그래프가 'final_comparison.png'로 저장되었습니다!")
plt.show()