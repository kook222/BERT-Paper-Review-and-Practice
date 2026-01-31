# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)
![Task](https://img.shields.io/badge/Task-Paper__Review_%26_Implementation-green)

<br>

## 👨‍💻 Author
**Park Seung Hyun**
* **Affiliation:** Pusan National University (PNU), PNU CLINK Lab
* **Email:** shp09240000@pusan.ac.kr
* **GitHub:** [kook222](https://github.com/kook222)

<br>
<hr>
<br>

## 📌 Project Overview
이 저장소는 **BERT (Devlin et al., 2018)** 논문을 깊이 있게 분석하고, 실제 코드 구현을 통해 그 성능을 검증한 스터디 기록입니다.

크게 두 가지 파트로 구성되어 있습니다:
1.  **Paper Review:** 논문의 핵심 아이디어(MLM, NSP, Architecture) 정리 및 발표 자료 제작.
2.  **Implementation:** IMDb 감성 분석 태스크를 통해 BERT와 GPT의 성능을 직접 비교 실험.

<br>

## 📚 Part 1. Paper Review (PDF)
제가 직접 작성하고 정리한 BERT 논문 분석 발표 자료입니다. 아래 링크를 클릭하면 전체 내용을 확인하실 수 있습니다.

> **[📄 발표 자료 보러가기 (Click to View PDF)](./BERT%20Pre-training%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding_paper_study.pdf)**

### 💡 Key Takeaways
* **Bidirectionality:** 기존 단방향(GPT)이나 Shallow Bidirectional(ELMo) 모델과 달리, 모든 레이어에서 양방향 문맥을 참조하는 **Deep Bidirectional** 구조를 제안.
* **Pre-training Tasks:**
    * **Masked LM (MLM):** 입력의 15%를 가리고 예측하며 문맥을 학습.
    * **Next Sentence Prediction (NSP):** 두 문장의 관계를 파악하는 능력 학습.
* **Feature-based vs Fine-tuning:** BERT는 Fine-tuning뿐만 아니라, 임베딩만 추출하여 사용하는 Feature-based 방식에서도 SOTA 급 성능을 보임.

<br>

## 📊 Part 2. Experimental Results (BERT vs GPT)
실제 IMDb 영화 리뷰 데이터셋(50k)을 사용하여 단방향 모델(OpenAI GPT)과 양방향 모델(BERT)**의 성능 차이를 검증했습니다.

### 1. Training Setup
* **Task:** Sentiment Analysis (Binary Classification)
* **Dataset:** IMDb Movie Reviews
* **Models:**
    * `OpenAI GPT` (110M params, Unidirectional)
    * `BERT-Base` (110M params, Bidirectional)
    * `BERT-Large` (340M params, Bidirectional)

### 2. Final Result Graph
**[실험 결과 요약]**
학습 진행(Epoch)에 따른 정확도(Accuracy) 변화 그래프입니다.

<p align="center">
  <img src="final_comparison.png" width="80%">
</p>

### 3. Quantitative Results (Accuracy)
| Model | Epoch 1 | Epoch 2 | Epoch 3 (Final) |
| :--- | :---: | :---: | :---: |
| **OpenAI GPT** | 87.9% | 87.9% | 89.1% |
| **BERT-Base** | 87.8% | 88.5% | 88.8% |
| **BERT-Large** | **88.8%** | **90.1%** | **89.9%** |

* **Observation:** BERT-Large 모델이 가장 높은 성능을 보였으나, 3 Epoch에서 과적합(Overfitting)으로 인해 성능이 소폭 하락함.

### 4. Analysis
* **BERT의 승리:** 동일한 파라미터 수(110M)를 가진 `BERT-Base`가 `OpenAI GPT`보다 높은 성능을 기록했습니다. 이는 감성 분석 태스크에서 **양방향 문맥 파악**이 얼마나 중요한지를 증명합니다.
* **Size Matters:** 모델 크기를 키운 `BERT-Large`는 압도적인 성능(약 89.9%)을 보여주었습니다.
* **Overfitting 이슈:** `BERT-Large`의 경우 3 Epoch에서 성능이 소폭 하락했는데, 이는 모델이 너무 강력하여 학습 데이터에 과적합(Overfitting)되기 시작했음을 시사합니다. (논문 권장 Epoch: 2~4회)

<br>

## 🚀 How to Run
본 프로젝트는 학습 단계별, 모델별로 코드가 분리되어 있어 직관적인 실행이 가능합니다.

### 1. Install Dependencies
필수 라이브러리를 설치합니다.

```bash
pip install -r requirements.txt

```

### 2. Training (Run Experiments)

원하는 모델의 스크립트를 직접 실행하여 학습을 시작합니다.

**A. Train BERT-Base**

```bash
# BERT 기본 모델 학습
python 1_train_bert.py

```

**B. Train BERT-Large**

```bash
# BERT Large 모델 학습
python 1_train_bert_large.py

```

**C. Train OpenAI-GPT**

```bash
# GPT 모델 학습 (비교군)
python 1_train_gpt.py

```

### 3. Inference & Testing

학습된 모델을 불러와 결과를 테스트합니다.

```bash
# BERT Base 예측 테스트
python 3_predict_bert.py

# BERT Large 예측 테스트
python 3_predict_bert_large.py

# GPT 예측 테스트
python 3_predict_gpt.py

```

### 4. Visualization

학습 로그(`training_history.json`)를 기반으로 최종 결과 그래프를 생성합니다.

```bash
python 4_final_graph.py

```

## 📂 Project Structure

이 프로젝트의 디렉토리 구조는 다음과 같습니다.

```bash
.
├── 1_train_bert.py        # BERT Base Training script
├── 1_train_bert_large.py  # BERT Large Training script
├── 1_train_gpt.py         # GPT Training script
├── 3_predict_bert.py      # BERT Base Inference script
├── 3_predict_bert_large.py # BERT Large Inference script
├── 3_predict_gpt.py       # GPT Inference script
├── 4_final_graph.py       # Result Visualization
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
└── ...

```

## 🛠 Tech Stack

* **Language:** Python 3.8+
* **Framework:** PyTorch, Hugging Face Transformers
* **Visualization:** Matplotlib, Seaborn

## 🔗 References

* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* [Improving Language Understanding by Generative Pre-Training (GPT)](https://www.google.com/search?q=https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
* [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
