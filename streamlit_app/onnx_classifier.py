import json
import os
import re

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_ID = os.environ.get("MODEL_ID", "lauraCeline/kinyarwanda-news-classification")
ONNX_FILE = "model_quantized.onnx"
MAX_LEN = 256


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _get_file(source, filename):
    if os.path.isdir(source):
        return os.path.join(source, filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(source, filename)


class Classifier:
    def __init__(self, source=MODEL_ID):
        self.tokenizer = Tokenizer.from_file(_get_file(source, "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_LEN)
        with open(_get_file(source, "config.json"), encoding="utf-8") as f:
            id2label = json.load(f)["id2label"]
        self.labels = [id2label[str(i)] for i in range(len(id2label))]
        self.session = ort.InferenceSession(
            _get_file(source, ONNX_FILE), providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

    def predict_proba(self, text):
        enc = self.tokenizer.encode(clean_text(text))
        ids = np.array([enc.ids], dtype=np.int64)
        feed = {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "token_type_ids": np.zeros_like(ids),
        }
        feed = {k: v for k, v in feed.items() if k in self.input_names}
        logits = self.session.run(None, feed)[0][0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return {label: float(p) for label, p in zip(self.labels, probs)}
