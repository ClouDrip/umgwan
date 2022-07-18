# KoBERT-nsmc

- KoBERT를 이용한 네이버 영화 리뷰 감정 분석 (sentiment classification)
- 🤗`Huggingface Tranformers`🤗 라이브러리를 이용하여 구현

## Dependencies

- torch==1.4.0
- transformers==2.10.0


## Usage

```bash
$ python3 main.py --model_type kobert --do_train --do_eval
```

## Prediction

```bash
$ python3 predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```

## Results

|                   | Accuracy (%) |
| ----------------- | ------------ |
| KoBERT            | **89.63**    |
| DistilKoBERT      | 88.41        |
| Bert-Multilingual | 87.07        |
| FastText          | 85.50        |

## References

- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [NSMC dataset](https://github.com/e9t/nsmc)
