from pathlib import Path

import pandas as pd
from datasets import load_dataset

BASE_DIR = Path(__file__).parent.parent

N_NORMAL = 1500   # câte exemple normale scurte adăugăm
MIN_WORDS = 3     # minim 3 cuvinte — evităm răspunsuri de un cuvânt
MAX_WORDS = 15    # maxim 15 cuvinte — mesaje scurte conversaționale

# descărcare
print("Încărcare dailydialog din fișier local...")
df = pd.read_parquet(BASE_DIR / "data" / "raw" / "dailydialog.parquet")
print(f"Conversații: {len(df)}")
print(f"Coloane: {df.columns.tolist()}")
print(df.iloc[0])

# extragere replici individuale
all_utterances = []
for _, row in df.iterrows():
    for utterance in row["utterances"]:
        utterance = utterance.strip()
        if utterance:
            all_utterances.append(utterance)

utterances_df = pd.DataFrame({"text": all_utterances})
print(f"Replici totale extrase: {len(utterances_df)}")

# filtrare după lungime
utterances_df["word_count"] = utterances_df["text"].str.split().str.len()
short_df = utterances_df[
    (utterances_df["word_count"] >= MIN_WORDS) &
    (utterances_df["word_count"] <= MAX_WORDS)
].copy()

print(f"Replici scurte ({MIN_WORDS}-{MAX_WORDS} cuvinte): {len(short_df)}")

# selecție aleatorie
selected_df = short_df.sample(
    min(N_NORMAL, len(short_df)),
    random_state=42
).copy()

print(f"Selectat: {len(selected_df)} exemple")

print("\n=== Exemple selectate ===")
for text in selected_df["text"].head(8).tolist():
    print(f"  [{len(text.split())} cuvinte] {text}")

# deduplicare față de dataset-ul existent
existing = pd.read_csv(BASE_DIR / "data" / "processed" / "full.csv")
existing_texts = set(existing["text"].str.lower().str.strip())

before = len(selected_df)
selected_df = selected_df[
    ~selected_df["text"].str.lower().str.strip().isin(existing_texts)
].copy()
print(f"\nDupă deduplicare: {len(selected_df)} (eliminat {before - len(selected_df)} duplicate)")

# format unificat — toate normale
selected_df["label_spam"]        = 0
selected_df["label_hate_speech"] = 0
selected_df["label_harassment"]  = 0
selected_df["label_obscene"]     = 0
selected_df["spam_subtype"]      = None
selected_df["hate_target"]       = None

selected_df = selected_df[[
    "text", "label_spam", "label_hate_speech",
    "label_harassment", "label_obscene",
    "spam_subtype", "hate_target"
]]

# split și integrare
shuffled = selected_df.sample(frac=1, random_state=42).reset_index(drop=True)
n       = len(shuffled)
n_test  = max(1, int(n * 0.15))
n_val   = max(1, int(n * 0.15))
n_train = n - n_test - n_val

dd_train = shuffled.iloc[:n_train].copy()
dd_val   = shuffled.iloc[n_train:n_train + n_val].copy()
dd_test  = shuffled.iloc[n_train + n_val:].copy()

dd_train["split"] = "train"
dd_val["split"]   = "val"
dd_test["split"]  = "test"

print(f"\n=== Distribuție split-uri daily_dialog ===")
for name, split in [("train", dd_train), ("val", dd_val), ("test", dd_test)]:
    print(f"{name:5}: {len(split):4} exemple normale scurte")

# combinare cu dataset-ul existent
train_existing = pd.read_csv(BASE_DIR / "data" / "processed" / "train_final.csv")
val_existing   = pd.read_csv(BASE_DIR / "data" / "processed" / "val_final.csv")
test_existing  = pd.read_csv(BASE_DIR / "data" / "processed" / "test_final.csv")


def align_and_concat(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    for col in existing.columns:
        if col not in new.columns:
            new[col] = 0 if "label" in col else None
    new = new[[c for c in existing.columns if c in new.columns]]
    return (
        pd.concat([existing, new], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


train_final = align_and_concat(train_existing, dd_train)
val_final   = align_and_concat(val_existing,   dd_val)
test_final  = align_and_concat(test_existing,  dd_test)

train_final.to_csv(BASE_DIR / "data" / "processed" / "train_final.csv", index=False)
val_final.to_csv(BASE_DIR / "data" / "processed" / "val_final.csv",     index=False)
test_final.to_csv(BASE_DIR / "data" / "processed" / "test_final.csv",   index=False)

print(f"\n=== Dataset final actualizat ===")
for name, split in [("train", train_final), ("val", val_final), ("test", test_final)]:
    spam = split["label_spam"].sum()
    hs   = split["label_hate_speech"].sum()
    har  = split["label_harassment"].sum()
    norm = ((split[["label_spam", "label_hate_speech", "label_harassment"]].sum(axis=1)) == 0).sum()
    print(f"{name:5}: {len(split):5} | spam: {spam} | hate: {hs:.0f} | harassment: {har:.0f} | normal: {norm}")

print("\n✓ train_final.csv / val_final.csv / test_final.csv actualizate")