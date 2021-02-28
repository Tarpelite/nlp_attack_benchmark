import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
import editdistance
from flask import Flask, request, jsonify
from flask_cors import CORS

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

task_to_keys = {
    "fake":("sentence1", None)
}


task_to_seq_lens = {
    "fake": 256
}

task_to_num_labels = {
    "fake": 128
}

model_path = ""
raw_test_file_path = ""
batch_size = 32
cache_dir = ""

UPLOAD_FOLDER = ""
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

config = AutoConfig.from_pretrained(
    model_path,
    num_labels = 2,
    cache_dir=cache_dir
)

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    use_fast = True
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    from_tf= False,
    config=config,
    cache_dir = cache_dir
)



label_to_id = {"1":0, "2":1}
data_files = {
    "test":raw_test_file_path
}

raw_test_datasets = load_dataset("csv", data_files = data_files)

def preprocess_function(examples):
    args = ("text")
    result["label"] = [label_to_id[l] for l in examples["label"]]
    return result

raw_test_datasets = datasets.map(preprocess_function, batched=True)
raw_test_dataset = raw_test_datasets["test"]

def compute_metrics(p:EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

data_collator = default_data_collator

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset = raw_test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

 

acc_score = 0.0

@app.route("/eval_raw", methods=["GET"])
def eval_raw():
    if acc_score > 0:
        return jsonify(
            {"raw_acc": acc_score}
        )
    else:
        eval_result = trainer.evaluate(eval_dataset=raw_test_dataset)
        acc_score = eval_result["accuracy"]
        return jsonify(
            "raw_acc":acc_score
        )

@app.route("eval_attack", method=["POST"])
def eval_attack():
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify(
                {"error":"No file part"}
            )
        file = request.files["file"]

        if file.filename == '':
            return jsonify(
                {"error": "No selected file"}
            )
        if file 












 





