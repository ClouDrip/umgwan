# 클라우드 기반 인공지능 모델을 활용한 악성 댓글 필터링 API 서비스

## Table of Contents
  1. [Members](#Members)
  2. [Project Overview](#Project-Overview)
  3. [Hardware](#Hardware)
  4. [Code Structure](#Code-Structure)

## Members

|                            김현욱                            |                            강진호                            |                            고유찬                            |                            이진구                            |                            김동규                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|[Github](https://github.com/powerwook) | [Github](https://github.com/jinho-Kang) | [Github](https://github.com/redronsean) | [Github](https://github.com/Leejingoo13) | [Github](https://github.com/TerryKimDK) |


## Project Overview
  * 개발 범위
    클라우드 기반 인공지능 모델 구현, 사용자 예시 웹사이트 구현
  * 목표
    1. 악성 댓글 필터링 인공지능 모델을 구현하고 클라우드 기반으로 사용자에게 API를 배포
    2. 사용자 예시 웹사이트를 구현하여 악성 댓글 필터링 모델의 성능을 비교/확인
  * Contributors
    * 김현욱: 악성 댓글 필터링 인공지능 모델 구현,  API 서버 구현
    * 강진호: 웹사이트 구현(토론 커뮤니티 웹사이트)
    * 고유찬: 웹사이트 구현(토론 커뮤니티 웹사이트)
    * 이진구: 웹사이트 구현(의류 쇼핑몰)
    * 김동규: 웹사이트 구현(의류 쇼핑몰)

## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-SXM2-32GB

## Code Structure
```text
├── code/                   
│   ├── crawl
│   │   └── bunjang_crawl.py
│   │
│   ├── multimodal-clf
│   │   ├── configs
│   │   │   ├── data/secondhad-goods.yaml
│   │   │   └── model/mobilenetv3_kluebert.yaml
│   │   ├── src
│   │   │   ├── augmentation
│   │   │   │   ├── methods.py
│   │   │   │   ├── policies.py
│   │   │   │   └── transforms.py
│   │   │   ├── utils
│   │   │   │   ├── common.py
│   │   │   │   └── data.py
│   │   │   ├── dataloader.py
│   │   │   ├── model.py
│   │   │   └── traniner.py
│   │   └── train.py
│   │   
│   ├── prototype
│   │   ├── models/mmclf
│   │   │   ├── best.pt
│   │   │   ├── config.yaml
│   │   │   ├── mmclf.py
│   │   │   ├── special_tokens_map.json
│   │   │   ├── tokenizer_config.json
│   │   │   ├── tokenizer.json
│   │   │   └── vocab.txt
│   │   ├── app.py
│   │   └── inference.py
│   │   
│   ├── text_extraction
│   │   ├── es_api.py
│   │   ├── make_vocab.py
│   │   └── text_extraction.py
│   │
│   ├── text_generation
│   │   ├── arguments.py
│   │   ├── data.py
│   │   ├── hashtag_preprocess.py
│   │   ├── inference.py
│   │   ├── preprocess.py
│   │   └── train.py                  
│   │
│   ├── requirements.txt
│   └── README.md
│
└── data/es_data                     
    └── vocab_space_ver2.txt                        
    
