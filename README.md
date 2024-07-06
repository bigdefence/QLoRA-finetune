# 🤖 한국어 AI 모델 파인튜닝 프로젝트

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)

한국어 AI 모델을 파인튜닝하고 테스트하기 위한 종합적인 프로젝트입니다. LLaMA 3 기반의 한국어 모델을 사용하여 특정 태스크에 맞게 모델을 조정하고 성능을 향상시킵니다.

## 📂 프로젝트 구조

```
.
├── fine_tune_test.py
├── merge_model_hf_upload.py
├── qlora_tune.py
└── fine_tunning_dataset.py
```

## 🚀 주요 기능

1. **모델 파인튜닝**: QLoRA 기법을 사용한 효율적인 파인튜닝
2. **모델 테스트**: 파인튜닝된 모델의 성능 평가
3. **모델 병합 및 업로드**: 파인튜닝된 모델을 병합하고 Hugging Face Hub에 업로드
4. **데이터셋 준비**: 커스텀 데이터셋을 Hugging Face 형식으로 변환 및 업로드

## 🛠️ 설치 방법

1. 저장소를 클론합니다:
   ```
   git clone https://github.com/your-username/korean-ai-model-finetuning.git
   ```

2. 필요한 패키지를 설치합니다:
   ```
   pip install -r requirements.txt
   ```

## 💻 사용 방법

### 모델 파인튜닝

```python
python qlora_tune.py
```

### 모델 테스트

```python
python fine_tune_test.py
```

### 모델 병합 및 업로드

```python
python merge_model_hf_upload.py
```

### 데이터셋 준비

```python
python fine_tunning_dataset.py
```

## 🔧 설정

각 스크립트 파일의 상단에 있는 변수들을 프로젝트에 맞게 수정하세요:

- `base_model`: 기본 모델의 Hugging Face 저장소 주소
- `new_model`: 파인튜닝된 모델을 저장할 경로
- `dataset_name`: 사용할 데이터셋의 Hugging Face 저장소 주소

## 👨‍💻 개발자 정보

- **이름**: 정강빈
- **버전**: 1.0.0

## 🤝 기여하기

프로젝트에 기여하고 싶으신가요? 다음 단계를 따라주세요:

1. 이 저장소를 포크합니다.
2. 새 브랜치를 생성합니다: `git checkout -b feature/AmazingFeature`
3. 변경사항을 커밋합니다: `git commit -m 'Add some AmazingFeature'`
4. 브랜치에 푸시합니다: `git push origin feature/AmazingFeature`
5. Pull Request를 생성합니다.

## 📜 라이선스

이 프로젝트는 [MIT 라이선스](https://opensource.org/licenses/MIT)하에 배포됩니다.

## 📞 연락처

프로젝트 관리자: [정강빈](mailto:bigdefence@naver.com)

프로젝트 링크: [https://github.com/bigdefence/QLoRA-finetune](https://github.com/bigdefence/QLoRA-finetune)
