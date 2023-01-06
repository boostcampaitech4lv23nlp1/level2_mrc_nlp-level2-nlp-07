# 부스트캠프 4기 NLP 07조 염보라
## Members
---

김한성|염성현|이재욱|최동민|홍인희|
:-:|:-:|:-:|:-:|:-:
<img src='https://user-images.githubusercontent.com/44632158/208237676-ae158236-16a5-4436-9a81-8e0727fe6412.jpeg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/44632158/208237686-c66a4f96-1be0-41e2-9fbf-3bf738796c1b.jpeg' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/108864803/208801820-5b050001-77ed-4714-acd2-3ad42c889ff2.png' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/108864803/208802208-0e227130-6fe5-4ca0-9226-46d2b07df9bf.png' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/97818356/208237742-7901464c-c4fc-4066-8a85-1488d56e0cce.jpg' height=80 width=80px></img>|
[Github](https://github.com/datakim1201)|[Github](https://github.com/neulvo)|[Github](https://github.com/datakim1201)|[Github](https://github.com/datakim1201)|[Github](https://github.com/datakim1201)
&nbsp;

## Wrap up report
[project report 바로가기](https://github.com/boostcampaitech4lv23nlp1/level2_klue_nlp-level2-nlp-07/blob/main/NLP%20%EA%B4%80%EA%B3%84%EC%B6%94%EC%B6%9C_NLP_%ED%8C%80%20%EB%A6%AC%ED%8F%AC%ED%8A%B8(07%EC%A1%B0).pdf)

&nbsp;

# Open-Domain Question Answering(ODQA)
## 프로젝트 수행 기간
>22/12/19 ~ 23/01/05

&nbsp;
## 프로젝트 개요
---
>**Open-Domain Question Answering (ODQA)** 은 주어지는 지문이 따로 존재하지 않고 사전에 구축되어있는 Knowledge resource 에서 질문에 대답할 수 있는 문서를 찾는 과정입니다. ODQA는 two-stage로 질문에 관련된 문서를 찾아주는 **“retriever”**, 관련된 문서를 읽고 적절한 답변을 찾아주는 **“reader”** 로 구성되어 있습니다. 두 가지 단계를 각각 구성하고 통합하여 어려운 질문을 던져도 답변을 해주는 ODQA 시스템을 만들고자 하였습니다.



<p align="center">
<img src ='https://user-images.githubusercontent.com/97818356/211024402-7702b8ed-1fd0-4278-9eff-de5597b7cd8b.jpg'></p>


&nbsp;

## 데이터 설명
---
아래는 제공하는 데이터셋의 분포를 보여줍니다.

<img src ='https://user-images.githubusercontent.com/97818356/211025102-b9ea49d1-40cc-49c7-9810-6c7424b7675e.jpg'>
데이터셋은 Huggingface 에서 제공하는 datasets를 이용하여 pyarrow 형식의 데이터로 저장되어있습니다. data 폴더의 구성은 아래와 같습니다.

```bash
./data/                        # 전체 데이터
    ./train_dataset/           # 학습에 사용할 데이터셋. train 과 validation 으로 구성 
    ./test_dataset/            # 제출에 사용될 데이터셋. validation 으로 구성 
    ./wikipedia_documents.json # 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.
```
data에 대한 argument 는 `code/arguments.py` 의 `DataTrainingArguments` 에서 확인 가능합니다.

&nbsp;
## 프로젝트 세부 내용
---
### Data
- EDA
- Preprocesisng
- Data Augmentation

### Retriever
- BM25
- Dense retriever
- Poly encoder

### Reader
- Scheduler & Optimizer

### Post processing
  
### Ensemble

&nbsp; 
## Train, Evaluation, Inference
---
```bash
# train, evaluation, inference를 통합하여 진행하고자 한다면, 아래 코드를 실행하세요.
python main.py
```
&nbsp;  
### Train, Evaluation
train, evaluation에서 필요한 세팅은 `code/config/config.yaml`에서 해주세요. 설정해줘야 할 값들은 아래와 같습니다.
```bash
# code/config/config.yaml
data:
    data_path: /opt/ml/input/data
    dataset_name: /opt/ml/input/data/train_dataset # train data
    aug_kor1: True                                 # kor 1.0 데이터 증강
    aug_kor2: True                                 # kor 2.1 데이터 증강
    aug_aihub: True                                # AI hub 데이터 증강
    overwrite_cache: False
    max_seq_length: 512
    pad_to_max_length: False
    doc_stride: 256
    max_answer_length: 30
    output_model_dir: ./models/model_output        # 파일 이름 변경하기

model:
    model_name: klue/roberta-large
    if_not_roberta: False           # bert는 True, roberta는 False
    huggingface_hub: True           # huggingface에 올리는건 True, 아니면 False

train:
    train_mode: True                # train 할 시, True로 설정
    seed: 42
    batch_size: 8
    epoch: 15
    lr : 1e-5
    weight_decay: 0.1
    warmup_ratio: 0.1
    logging_step: 4000
    eval_step: 4000                 # eval step와 save step은 같이 바뀝니다.
    label_smoothing_factor: 0.1
    load_best_model_at_end: True
    gradient_accumulation_steps: 1
    optimizer_step_size: 8000       # 본인이 하는거에 맞춰서 주기 5 ~ 10정도에 맞게 설정하기

wandb:
    wandb_mode: True                # train 할 시, wandb 사용 유무 설정
    entity: mrc_bora
    project_name: testtest
    exp_name: robeta-large_kor1-2_aihub
```
```bash
# train, evaluation만 한다면
# train_mode = True, test_mode = False 설정
python main.py
```
&nbsp; 
### Inference
inference에서 필요한 세팅은 `code/config/config.yaml`에서 해주세요. 설정해줘야 할 값들은 아래와 같습니다.
```bash
data:
    data_path: /opt/ml/input/data
    context_path: wikipedia_documents.json       # 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.
    test_dataset_name: /opt/ml/input/data/test_dataset/  # 제출에 사용될 데이터셋. validation 으로 구성 
    overwrite_cache: False
    max_answer_length: 30
    eval_retrieval: True
    dense_retrieval: False              # dense retrieval 사용 시 True, sparse retrieval 사용 시 False
    num_clusters: 64
    top_k_retrieval: 40
    use_faiss: False                    # BM25의 경우 faiss를 사용하지 않습니다.
    output_model_dir: ./models/model_output     # 학습된 모델 경로 
    output_json_dir: ./outputs/output_pred/     # output predictions

model:
    model_name: klue/roberta-large      # 학습에 사용한 PLM name
    if_not_roberta: False               # bert는 True, roberta는 False

test:
    test_mode: True                     # inference 할 시, True로 설정
    BM25: True                          # BM25는 True, TF-IDF는 False

encoder:                                # dense encoder 설정
    model_name: klue/roberta-base
    epoch: 50
    batch_size: 16
    lr: 2e-5
    weight_decay: 0.01
    dense_train: False                  # encoder 학습 시, True
    faiss_gpu: False                    # faiss GPU 사용 시, True
    embedding_name: dense_embedding.bin
    encoder_postfix: test               # if dense_train == True
    load_encoder_path: test             # if dense_train == False
```
```bash
# inference만 한다면
# train_mode = False, test_mode = True 설정
python main.py
```

### How to submit
inference 후, `output_json_dir` 위치에 `predictions.json`이라는 파일이 생성됩니다. 해당 파일을 제출해주세요.

&nbsp; 
## 프로젝트 구조
---
```
ODQA Project/
│
├── baseline/ 
├── code/ 
│   ├── config/
│   │   ├── config.yaml
│   │   └── sweep_config.yaml
│   │
│   ├── install/
│   │   └── install_requirements.sh
│   │
│   ├── notebook/
│   │   ├── EDA_for_wikipedia.ipynb
│   │   ├── ...
│   │   └── squad.ipynb
│   │
│   ├── arguments.py
│   ├── bm25.py
│   ├── dense_retrieval.py
│   ├── ensemble.py
│   ├── inference.py
│   ├── main.py
│   ├── retrieval_model.py
│   ├── sparse_retrieval.py
│   ├── train.py
│   ├── train_sweep.py
│   ├── trainer_qa.py
│   └── utils_qa.py
│
├── .gitignore
├── README.md
│
└── thanks for comming I'm Yeombora
```