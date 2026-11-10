# ============================================================
# BLOC 1 — încărcare date
# ============================================================

import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.model.labels import LABEL_COLS

def load_split(path: str) -> tuple[list[str], list[list[int]]]:
    df = pd.read_csv(path)

    # verificare că toate coloanele necesare există
    missing = [c for c in LABEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Coloane lipsă în {path}: {missing}")

    texts  = df["text"].fillna("").tolist()
    labels = df[LABEL_COLS].fillna(0).astype(int).values.tolist()

    return texts, labels

# ============================================================
# BLOC 2 — tokenizare și dataset
# ============================================================

import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast

MODEL_NAME = "distilbert/distilbert-base-multilingual-cased"
MAX_LENGTH = 128  # lungimea maximă a secvenței în tokeni

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

class HateSpamDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[list[int]]):
        self.texts  = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text  = self.texts[idx]
        label = self.labels[idx]

        # tokenizare cu padding și truncare la MAX_LENGTH
        encoding = tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,        # taie textele mai lungi de MAX_LENGTH
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),

            "labels": torch.FloatTensor(label),
        }


# ============================================================
# BLOC 3 — definirea modelului
# ============================================================

from torch import nn
from transformers import DistilBertModel

NUM_LABELS = len(LABEL_COLS)  # 4


class HateSpamClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.3):
        super().__init__()

        # backbone preantrenat - greutățile se descarcă automat
        self.distilbert = DistilBertModel.from_pretrained(model_name)

        # strat de regularizare - dezactivează aleator 30% din neuroni
        self.dropout = nn.Dropout(dropout)

        # capul de clasificare - singurul strat complet nou
        # 768 = dimensiunea vectorului CLS din DistilBERT
        self.classifier = nn.Linear(768, num_labels)

    # pasul de propagare înainte (forward pass)
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # DistilBERT returnează reprezentările pentru toți tokenii
        # last_hidden_state: [batch_size, seq_len, 768]
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_output = outputs.last_hidden_state[:, 0, :]

        # aplicăm dropout
        cls_output = self.dropout(cls_output)

        # calculăm logits
        # logits: [batch_size, 4]
        logits = self.classifier(cls_output)

        return logits


# ============================================================
# BLOC 4 — Antrenament
# ============================================================

from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

# hiperparametri

BATCH_SIZE    = 16      # câte exemple procesăm simultan

NUM_EPOCHS    = 3       # câte treceri complete prin dataset

LEARNING_RATE = 2e-5    # cât de mult se modifică greutățile la fiecare pas

WARMUP_RATIO  = 0.1     # primii 10% din pași

def train_epoch(
    model:      HateSpamClassifier,
    loader:     DataLoader,
    optimizer:  AdamW,
    scheduler,
    loss_fn:    nn.BCEWithLogitsLoss,
    device:     torch.device,
    epoch:      int,
) -> float:

    model.train()  # activează dropout
    total_loss = 0.0

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # resetăm gradienții acumulați din pasul anterior
        optimizer.zero_grad()

        # forward pass
        logits = model(input_ids, attention_mask)

        # calculăm loss-ul
        loss = loss_fn(logits, labels)

        # backward pass (propagare înapoi)
        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # actualizăm greutățile
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # progres la fiecare 100 de pași
        if (step + 1) % 100 == 0:
            avg = total_loss / (step + 1)
            print(f"  Epoch {epoch} | Step {step+1}/{len(loader)} | Loss: {avg:.4f}")

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(
    model:   HateSpamClassifier,
    loader:  DataLoader,
    loss_fn: nn.BCEWithLogitsLoss,
    device:  torch.device,
) -> tuple[float, dict]:
    from sklearn.metrics import f1_score
    import numpy as np

    model.eval()  # dezactivează dropout
    total_loss  = 0.0
    all_preds   = []
    all_labels  = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss   = loss_fn(logits, labels)
        total_loss += loss.item()

        # aplicăm sigmoid și prag 0.5 pentru decizii binare
        preds = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.int().cpu().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # F1 macro per etichetă
    f1_scores = {}
    for i, col in enumerate(LABEL_COLS):
        f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        f1_scores[col] = round(f1, 4)

    return total_loss / len(loader), f1_scores


def run_training(
    train_texts:  list[str],
    train_labels: list[list[int]],
    val_texts:    list[str],
    val_labels:   list[list[int]],
) -> HateSpamClassifier:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # dataset-uri și loadere
    train_dataset = HateSpamDataset(train_texts, train_labels)
    val_dataset   = HateSpamDataset(val_texts,   val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,          # amestecăm la fiecare epocă
        num_workers=0,         # 0 = fără multiprocessing (mai simplu pe Windows)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # model
    model = HateSpamClassifier(MODEL_NAME, NUM_LABELS).to(device)

    # class weights
    class_weights = torch.FloatTensor([20.0, 3.98, 4.39, 30.0]).to(device)

    # loss function cu class weights
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # optimizer AdamW
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # scheduler cu warmup linear
    total_steps  = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"Total pași antrenament : {total_steps}")
    print(f"Warmup pași            : {warmup_steps}")

    best_val_f1  = 0.0
    best_epoch   = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch}/{NUM_EPOCHS}")
        print(f"{'='*50}")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device, epoch
        )
        val_loss, val_f1 = evaluate(model, val_loader, loss_fn, device)

        avg_f1 = sum(val_f1.values()) / len(val_f1)

        print(f"\nEpoch {epoch} — Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val F1 per etichetă:")
        for col, f1 in val_f1.items():
            print(f"  {col:<25}: {f1:.4f}")
        print(f"Val F1 mediu: {avg_f1:.4f}")

        # salvăm cel mai bun model după F1 mediu pe val
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1
            best_epoch  = epoch
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"  ✓ Model salvat (best val F1: {best_val_f1:.4f})")

    print(f"\nAntrenament finalizat. Cel mai bun model: epoch {best_epoch} (F1={best_val_f1:.4f})")
    return model

# ============================================================
# BLOC 5 — Evaluare finală și salvare
# ============================================================

def evaluate_final(
    model:       HateSpamClassifier,
    test_texts:  list[str],
    test_labels: list[list[int]],
    device:      torch.device,
    model_path:  str = "models/best_model.pt",
) -> None:

    from sklearn.metrics import classification_report
    import numpy as np

    # încărcăm cel mai bun model
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.to(device)
    model.eval()
    print(f"\nModel încărcat din {model_path}")

    test_dataset = HateSpamDataset(test_texts, test_labels)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]

            logits = model(input_ids, attention_mask)
            preds  = (torch.sigmoid(logits) >= 0.5).int().cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.int().numpy())

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    print("\n=== Evaluare finală pe TEST SET ===\n")
    for i, col in enumerate(LABEL_COLS):
        report = classification_report(
            all_labels[:, i],
            all_preds[:, i],
            labels=[0, 1],
            target_names=["negative", "positive"],
            zero_division=0,
        )
        print(f"--- {col} ---")
        print(report)


def save_artifacts(model: HateSpamClassifier) -> None:
    import os
    os.makedirs("models/tokenizer", exist_ok=True)
    tokenizer.save_pretrained("models/tokenizer/")
    print("✓ Tokenizer salvat în models/tokenizer/")
    print("✓ Model salvat în models/best_model.pt")
    print("\nFișiere necesare pentru inferență:")
    print("  models/best_model.pt")
    print("  models/tokenizer/")


# --- test manual ---
if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)

    train_texts, train_labels = load_split("data/processed/train_augmented.csv")
    val_texts,   val_labels   = load_split("data/processed/val_final.csv")
    test_texts,  test_labels  = load_split("data/processed/test_final.csv")

    print(f"Train : {len(train_texts)} exemple")
    print(f"Val   : {len(val_texts)} exemple")
    print(f"Test  : {len(test_texts)} exemple")

    # antrenament
    model = run_training(train_texts, train_labels, val_texts, val_labels)

    # evaluare finală pe test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_final(model, test_texts, test_labels, device)

    # salvare tokenizer
    save_artifacts(model)