import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
# încărcare

df = pd.read_csv(
    BASE_DIR / "data" / "raw" / "sms_spam_india.csv",
    encoding="utf-8",
    on_bad_lines="skip",
)

print(f"Încărcat: {len(df)} rânduri")
print(f"Coloane: {df.columns.tolist()}")
print(f"\nDistribuție etichete brute:")
print(df["Label"].value_counts())

# curățare
# redenumim coloanele la standardul proiectului
df = df.rename(columns={"Msg": "text", "Label": "label"})

# normalizăm etichetele
df["label"] = df["label"].str.strip().str.lower()

# eliminăm etichete necunoscute
df = df[df["label"].isin({"ham", "spam"})].copy()

# curățăm textul
df["text"] = df["text"].fillna("").astype(str).str.strip()

# fix encoding corupt 
def fix_encoding(text: str) -> str:
    try:
        fixed = text.encode("latin-1").decode("utf-8")
        return fixed
    except (UnicodeDecodeError, UnicodeEncodeError):
        return re.sub(r'[^\x00-\x7F\u0080-\u024F]+', ' ', text)

df["text"] = df["text"].apply(fix_encoding)
df["text"] = df["text"].str.replace(r'\s+', ' ', regex=True).str.strip()

# eliminăm textele prea scurte (sub 3 caractere)
df = df[df["text"].str.len() >= 3].copy()
print(f"\nDupă curățare: {len(df)} rânduri")

# deduplicare față de dataset
existing = pd.read_csv(BASE_DIR / "data" / "processed" / "full.csv")
existing_texts = set(existing["text"].str.lower().str.strip())

before = len(df)
df = df[~df["text"].str.lower().str.strip().isin(existing_texts)].copy()
print(f"După deduplicare: {len(df)} rânduri (eliminat {before - len(df)} duplicate)")

# format unificat
df["label_spam"]        = (df["label"] == "spam").astype(int)
df["label_hate_speech"] = 0
df["label_harassment"]  = 0
df["label_obscene"]     = 0
df["hate_target"]       = None

# spam_subtype prin keyword-based
PHISHING_KEYWORDS = [
    "verify", "account", "bank", "password", "login",
    "confirm", "credentials", "secure", "suspended",
    "otp", "kyc", "click here to activate",
]

SCAM_KEYWORDS = [
    "won", "winner", "prize", "award", "guaranteed",
    "claim", "cash", "bitcoin", "crypto", "selected to receive",
    "call to claim", "final notice", "profit", "investment",
]

def assign_subtype(row) -> str | None:
    if row["label_spam"] == 0:
        return None
    text_lower = row["text"].lower()
    if any(kw in text_lower for kw in PHISHING_KEYWORDS):
        return "phishing"
    if any(kw in text_lower for kw in SCAM_KEYWORDS):
        return "scam"
    return "junk"

df["spam_subtype"] = df.apply(assign_subtype, axis=1)

# păstrăm coloanele necesare
df = df[["text", "label_spam", "label_hate_speech",
         "label_harassment", "label_obscene",
         "spam_subtype", "hate_target"]]

print(f"\n=== Distribuție finală ===")
print(f"ham (normal): {(df['label_spam'] == 0).sum()}")
print(f"spam total  : {(df['label_spam'] == 1).sum()}")
print(f"\nDistribuție spam_subtype:")
print(df[df['label_spam'] == 1]['spam_subtype'].value_counts())

# split și integrare în dataset-ul existent
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
n           = len(df_shuffled)
n_test      = max(1, int(n * 0.15))
n_val       = max(1, int(n * 0.15))
n_train     = n - n_test - n_val

india_train = df_shuffled.iloc[:n_train].copy()
india_val   = df_shuffled.iloc[n_train:n_train + n_val].copy()
india_test  = df_shuffled.iloc[n_train + n_val:].copy()

india_train["split"] = "train"
india_val["split"]   = "val"
india_test["split"]  = "test"

print(f"\n=== Distribuție split-uri India ===")
for name, split in [("train", india_train), ("val", india_val), ("test", india_test)]:
    spam = split["label_spam"].sum()
    print(f"{name:5}: {len(split):4} | spam: {spam} ({spam/len(split):.1%})")

# combinare cu dataset
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

train_final = align_and_concat(train_existing, india_train)
val_final   = align_and_concat(val_existing,   india_val)
test_final  = align_and_concat(test_existing,  india_test)

train_final.to_csv(BASE_DIR / "data" / "processed" / "train_final.csv", index=False)
val_final.to_csv(BASE_DIR / "data" / "processed" / "val_final.csv",     index=False)
test_final.to_csv(BASE_DIR / "data" / "processed" / "test_final.csv",   index=False)

print(f"\n=== Dataset final actualizat ===")
for name, split in [("train", train_final), ("val", val_final), ("test", test_final)]:
    spam = split["label_spam"].sum()
    print(f"{name:5}: {len(split):5} | spam: {spam} ({spam/len(split):.1%})")

print("\n✓ train_final.csv / val_final.csv / test_final.csv actualizate")