from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast

from huggingface_hub import hf_hub_download

from app.model.labels import (
    INTENTS,
    LABEL_COLS,
    score_to_severity,
)

# paths
_MODELS_DIR    = Path(__file__).parent.parent.parent / "models"
_MODEL_PATH    = _MODELS_DIR / "best_model.pt"
_TOKENIZER_DIR = _MODELS_DIR / "tokenizer"

# arhitectura modelului
NUM_LABELS = len(LABEL_COLS)  # 4
MAX_LENGTH = 128

# link to Hugging Face Hub repository for the model
_REPO_ID = "cnegru38/haitedetector-distilbert"

def _ensure_model() -> Path:
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODELS_DIR.mkdir(exist_ok=True)
    print("Downloading model weights (~514 MB), first run only...")
    path = hf_hub_download(
        repo_id=_REPO_ID,
        filename="best_model.pt",
        local_dir=_MODELS_DIR,
    )
    return Path(path)

class HateSpamClassifier(nn.Module):

    def __init__(self, num_labels: int = NUM_LABELS, dropout: float = 0.3):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(
            "distilbert/distilbert-base-multilingual-cased"
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs    = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)

# model și tokenizer încărcate o singură dată la import
def _load_model() -> tuple[HateSpamClassifier, DistilBertTokenizerFast]:
    model_path = _ensure_model()

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(_TOKENIZER_DIR))

    model = HateSpamClassifier()
    state = torch.load(str(model_path), map_location="cpu")
    result = model.load_state_dict(state, strict=True)
    print(f"Missing keys    : {result.missing_keys}")
    print(f"Unexpected keys : {result.unexpected_keys}")
    model.eval()
    return model, tokenizer

_model, _tokenizer = _load_model()

@dataclass
class InferenceResult:
    scores:        dict[str, float] = field(default_factory=dict)
    decisions:     dict[str, bool]  = field(default_factory=dict)
    spam_subtype:  str | None       = None
    hate_target:   str | None       = None
    hate_severity: str | None       = None

# inferență principală
def _get_scores(text: str) -> dict[str, float]:
    encoding = _tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = _model(
            input_ids      = encoding["input_ids"],
            attention_mask = encoding["attention_mask"],
        )

    probs = torch.sigmoid(logits).squeeze(0).tolist()

    return {
    label: round(prob, 4)
    for label, prob in zip(INTENTS, probs)
    }

# clasificări secundare

# scoruri minime pentru clasificările secundare
_SUBTYPE_THRESHOLD = 0.35
_TARGET_THRESHOLD  = 0.35


def _get_spam_subtype(text: str) -> str | None:
    text_lower = text.lower()

    phishing_kw = ["verify", "account", "bank", "password", "login",
                   "confirm", "credentials", "secure", "suspended"]
    scam_kw     = ["won", "winner", "prize", "award", "guaranteed",
                   "claim", "cash", "bitcoin", "crypto", "inheritance"]

    if any(kw in text_lower for kw in phishing_kw):
        return "phishing"
    if any(kw in text_lower for kw in scam_kw):
        return "scam"
    return "junk"

def _get_hate_target(text: str) -> str:
    text_lower = text.lower()

    targets = {
        "race":        ["black", "white", "asian", "african", "race", "racial"],
        "religion":    ["muslim", "jewish", "christian", "islam", "religion", "mosque"],
        "gender":      ["women", "woman", "men", "female", "male", "gender"],
        "orientation": ["gay", "lesbian", "homosexual", "queer", "lgbt"],
        "ethnicity":   ["immigrant", "refugee", "hispanic", "latino", "ethnic"],
    }

    for target, keywords in targets.items():
        if any(kw in text_lower for kw in keywords):
            return target
    return "other"

# interfața publică
def run_inference(
    text:          str,
    thresholds:    dict[str, float],
    thresholds_ro: dict[str, float] | None = None,
    lang:          str = "en",
) -> InferenceResult:

    # etapa 1: alegere praguri bazat pe limbă
    # dacă thresholds_ro nu e furnizat sau limba nu e RO, folosim EN
    active_thresholds = (
        thresholds_ro
        if lang == "ro" and thresholds_ro is not None
        else thresholds
    )

    # etapa 2: scoruri
    scores = _get_scores(text)

    # etapa 3: decizii pe baza pragurilor active
    decisions = {
        label: scores.get(label, 0.0) >= active_thresholds.get(label, 0.5)
        for label in INTENTS
    }

    result = InferenceResult(scores=scores, decisions=decisions)

    # etapa 4 & 5: clasificări secundare condiționate
    if decisions.get("spam"):
        result.spam_subtype = _get_spam_subtype(text)

    if decisions.get("hate_speech"):
        result.hate_target   = _get_hate_target(text)
        result.hate_severity = score_to_severity(scores.get("hate_speech", 0.0))

    return result

# test manual

if __name__ == "__main__":
    from app.core.taxonomy_loader import load_taxonomy
    from pathlib import Path

    taxonomy = load_taxonomy(
        Path(__file__).parent.parent.parent / "taxonomy.yaml"
    )

    examples = [
        ("Free entry! Win £1000 prize, call now! URGENT 🔥💰",  "spam scam"),
        ("Verify your bank account immediately or it will be suspended.", "spam phishing"),
        ("I hate all people from that country, they should leave.",       "hate speech"),
        ("you are an idiot go away",                               "harassment"),
        ("What a beautiful day outside today!",                           "normal"),
        ("Du-te naibii, ești un prost!",                                 "harassment RO"),
    ]

    print(f"{'TEXT':<55} {'DECIZII':<30} {'SCOR HS'}")
    print("-" * 100)
    from app.model.preprocess import preprocess

    for text, expected in examples:
        clean_text, lang = preprocess(text)
        r = run_inference(clean_text, taxonomy.thresholds, taxonomy.thresholds_ro, lang)
        active = [k.replace("label_", "") for k, v in r.decisions.items() if v]
        hs_score = r.scores.get("label_hate_speech", 0)
        print(f"{text[:54]:<55} {str(active):<30} {hs_score:.3f}   # expected: {expected}")