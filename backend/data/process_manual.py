# data/process_manual.py
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent.parent

COLUMNS = [
    "text", "label_spam", "label_hate_speech",
    "label_harassment", "label_obscene",
    "spam_subtype", "hate_target",
]

normal_df = pd.read_csv(
    BASE_DIR / "data" / "raw" / "manual_normal.csv",
    header=0,
)
spam_df = pd.read_csv(
    BASE_DIR / "data" / "raw" / "manual_spam.csv",
    header=0,
)

print(f"Exemple normale  : {len(normal_df)}")
print(f"Exemple spam     : {len(spam_df)}")

new_df = pd.concat([normal_df, spam_df], ignore_index=True)

for col in ["label_spam", "label_hate_speech", "label_harassment", "label_obscene"]:
    new_df[col] = new_df[col].fillna(0).astype(int)

new_df["text"] = new_df["text"].astype(str).str.strip()
new_df = new_df[new_df["text"].str.len() > 0].copy()

train_path = BASE_DIR / "data" / "processed" / "train_final.csv"
train_df   = pd.read_csv(train_path)

existing_texts = set(train_df["text"].str.lower().str.strip())
before = len(new_df)
new_df = new_df[~new_df["text"].str.lower().str.strip().isin(existing_texts)].copy()
print(f"După deduplicare: {len(new_df)} (eliminat {before - len(new_df)} duplicate)")

new_df["split"] = "train"
for col in train_df.columns:
    if col not in new_df.columns:
        new_df[col] = 0 if "label" in col else None
new_df = new_df[[c for c in train_df.columns if c in new_df.columns]]

train_updated = pd.concat([train_df, new_df], ignore_index=True)\
                  .sample(frac=1, random_state=42).reset_index(drop=True)
train_updated.to_csv(train_path, index=False)

print(f"\nTotal  : {len(train_updated)}")
print(f"spam   : {train_updated['label_spam'].sum()}")
print(f"✓ train_final.csv actualizat cu {len(new_df)} exemple noi")