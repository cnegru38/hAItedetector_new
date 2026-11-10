import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# încărcare
df = pd.read_csv(
    BASE_DIR / "data" / "raw" / "mendeley_spam.csv",
    header=None,
    names=["label", "text", "url", "email", "phone"],
    encoding="utf-8",
    on_bad_lines="skip",   # sărim rândurile corupte
)

print(f"Încărcat: {len(df)} rânduri")
print(f"\nDistribuție etichete brute:")
print(df["label"].value_counts())

# curățare
# normalizăm etichetele
df["label"] = df["label"].str.strip().str.lower()

# eliminăm rândurile cu etichetă necunoscută
valid_labels = {"ham", "spam", "smishing"}
df = df[df["label"].isin(valid_labels)].copy()
print(f"\nDupă filtrare etichete valide: {len(df)} rânduri")

# curățăm textul
df["text"] = df["text"].astype(str).str.strip()

# eliminăm caracterele corupte din conversie PDF-text
df["text"] = df["text"].apply(lambda t: re.sub(r'[^\x00-\x7F\u0080-\u024F]+', ' ', t))
df["text"] = df["text"].str.replace(r'\s+', ' ', regex=True).str.strip()

# eliminăm textele goale sau prea scurte (sub 3 caractere)
df = df[df["text"].str.len() >= 3].copy()
print(f"După curățare text: {len(df)} rânduri")

# deduplicare față de sms spam collection
existing = pd.read_csv(BASE_DIR / "data" / "processed" / "full.csv")
existing_texts = set(existing["text"].str.lower().str.strip())

before = len(df)
df = df[~df["text"].str.lower().str.strip().isin(existing_texts)].copy()
print(f"După deduplicare față de dataset existent: {len(df)} rânduri (eliminat {before - len(df)} duplicate)")

# format unificat
def map_label(label: str) -> dict:
    if label == "ham":
        return {
            "label_spam": 0,
            "label_hate_speech": 0,
            "label_harassment": 0,
            "label_obscene": 0,
            "spam_subtype": None,
        }
    elif label == "smishing":
        return {
            "label_spam": 1,
            "label_hate_speech": 0,
            "label_harassment": 0,
            "label_obscene": 0,
            "spam_subtype": "phishing",
        }
    else:  # spam
        return {
            "label_spam": 1,
            "label_hate_speech": 0,
            "label_harassment": 0,
            "label_obscene": 0,
            "spam_subtype": "junk",
        }

# aplicăm maparea
mapped = df["label"].apply(map_label)
df = df.join(pd.DataFrame(mapped.tolist()))

SCAM_KEYWORDS = [
    "won", "winner", "prize", "award", "guaranteed", "claim",
    "cash", "bitcoin", "crypto", "inheritance", "selected to receive",
    "call to claim", "final notice",
]

def refine_subtype(row):
    if row["label_spam"] == 0 or row["spam_subtype"] == "phishing":
        return row["spam_subtype"]
    text_lower = row["text"].lower()
    if any(kw in text_lower for kw in SCAM_KEYWORDS):
        return "scam"
    return "junk"

df["spam_subtype"] = df.apply(refine_subtype, axis=1)

# păstrăm doar coloanele necesare
df = df[["text", "label_spam", "label_hate_speech",
         "label_harassment", "label_obscene", "spam_subtype"]]

# adăugăm hate_target
df["hate_target"] = None

print(f"\n=== Distribuție finală ===")
print(f"ham (normal): {(df['label_spam'] == 0).sum()}")
print(f"spam total  : {(df['label_spam'] == 1).sum()}")
print(f"\nDistribuție spam_subtype:")
print(df[df['label_spam'] == 1]['spam_subtype'].value_counts())

# adăugăm direct la split-urile existente
train_existing = pd.read_csv(BASE_DIR / "data" / "processed" / "train_final.csv")
val_existing   = pd.read_csv(BASE_DIR / "data" / "processed" / "val_final.csv")
test_existing  = pd.read_csv(BASE_DIR / "data" / "processed" / "test_final.csv")

# fix NaN
df["label_spam"] = df["label_spam"].fillna(0).astype(int)

# split manual
n = len(df)
n_test  = max(1, int(n * 0.15))
n_val   = max(1, int(n * 0.15))
n_train = n - n_test - n_val

df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
mendeley_train = df_shuffled.iloc[:n_train].copy()
mendeley_val   = df_shuffled.iloc[n_train:n_train+n_val].copy()
mendeley_test  = df_shuffled.iloc[n_train+n_val:].copy()

mendeley_train["split"] = "train"
mendeley_val["split"]   = "val"
mendeley_test["split"]  = "test"

print(f"\n=== Distribuție Mendeley split-uri ===")
for name, split in [("train", mendeley_train), ("val", mendeley_val), ("test", mendeley_test)]:
    spam = split["label_spam"].sum()
    print(f"{name:5}: {len(split):3} | spam: {spam}")

# combinăm cu existentul și salvăm
def align_and_concat(existing, new):
    for col in existing.columns:
        if col not in new.columns:
            new[col] = 0 if "label" in col else None
    new = new[[c for c in existing.columns if c in new.columns]]
    return pd.concat([existing, new], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

train_final = align_and_concat(train_existing, mendeley_train)
val_final   = align_and_concat(val_existing,   mendeley_val)
test_final  = align_and_concat(test_existing,  mendeley_test)

train_final.to_csv(BASE_DIR / "data" / "processed" / "train_final.csv", index=False)
val_final.to_csv(BASE_DIR / "data" / "processed" / "val_final.csv",     index=False)
test_final.to_csv(BASE_DIR / "data" / "processed" / "test_final.csv",   index=False)

print(f"\n=== Dataset final actualizat ===")
for name, split in [("train", train_final), ("val", val_final), ("test", test_final)]:
    spam = split["label_spam"].sum()
    print(f"{name:5}: {len(split):5} | spam: {spam} ({spam/len(split):.1%})")

print("\n✓ train_final.csv / val_final.csv / test_final.csv actualizate")