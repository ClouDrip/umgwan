import os
import logging
import argparse
from tqdm import tqdm, trange
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForSequenceClassification

from utils import init_logger, load_tokenizer

import uvicorn
from fastapi import FastAPI
import datetime
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from starlette.responses import JSONResponse

app = FastAPI()

@app.get(path='/', description="health check 포인트")
def health_check():
    return "OK"

class InItem(BaseModel):
    id: int
    document: str

class InList(BaseModel):
    bell_in: List[InItem]

def get_id(item: InItem):
    return item.id

def get_document(item: InItem):
    return item.document

def process_in_list(bell_in: InList):
    pass

logger = logging.getLogger(__name__)


def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, 'training_args.bin'))


def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)  # Config will be automatically loaded from model_dir
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


def convert_input_file_to_tensor_dataset(input_json, pred_config,
                                         args,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    tokenizer = load_tokenizer(args)
    tok_input_json = input_json

    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    print(len(tok_input_json['bell_in']))
    for idx in range(len(tok_input_json['bell_in'])):
        line = get_document(tok_input_json['bell_in'][idx]).strip()
        tokens = tokenizer.tokenize(line)
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[:(args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)

    # with open(pred_config.input_file, "r", encoding="utf-8") as f:
    #     for line in f:
    #         line = line.strip()
    #         tokens = tokenizer.tokenize(line)
    #         # Account for [CLS] and [SEP]
    #         special_tokens_count = 2
    #         if len(tokens) > args.max_seq_len - special_tokens_count:
    #             tokens = tokens[:(args.max_seq_len - special_tokens_count)]

    #         # Add [SEP] token
    #         tokens += [sep_token]
    #         token_type_ids = [sequence_a_segment_id] * len(tokens)

    #         # Add [CLS] token
    #         tokens = [cls_token] + tokens
    #         token_type_ids = [cls_token_segment_id] + token_type_ids

    #         input_ids = tokenizer.convert_tokens_to_ids(tokens)

    #         # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
    #         attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    #         # Zero-pad up to the sequence length.
    #         padding_length = args.max_seq_len - len(input_ids)
    #         input_ids = input_ids + ([pad_token_id] * padding_length)
    #         attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    #         token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    #         all_input_ids.append(input_ids)
    #         all_attention_mask.append(attention_mask)
    #         all_token_type_ids.append(token_type_ids)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)

    return dataset


def predict(pred_config, in_json):
    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)
    model = load_model(pred_config, args, device)
    logger.info(args)
    input_json = in_json

    # Convert input file to TensorDataset
    dataset = convert_input_file_to_tensor_dataset(input_json, pred_config, args)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}
            if args.model_type != "distilkobert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    preds = np.argmax(preds, axis=1)
    print(preds)

    output_json = dict(bell_out = [])
    for pred,idx in zip(preds, range(len(in_json['bell_in']))):
        output_json['bell_out'].append(dict(id = get_id(input_json['bell_in'][idx]), label = int(pred)))
    return output_json

    # Write to output file
    # with open(pred_config.output_file, "w", encoding="utf-8") as f:
    #     for pred in preds:
    #         f.write(f"{pred}\n")

    logger.info("Prediction Done!")

@app.post("/")  # 요청 url 지정
async def pred_conts(bell_in: InList):
    try :
        in_json = dict(bell_in)
        
    except Exception as e :
        print(e,'ERROR')
    return JSONResponse(predict(pred_config, in_json))

if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="bell_in.json", type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default="bell_out.json", type=str, help="Output file for prediction")
    parser.add_argument("--model_dir", default="./model/kcelectra-base", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()
    uvicorn.run(app, host='0.0.0.0')