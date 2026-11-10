import re
import pandas as pd
from sklearn.model_selection import train_test_split

# încărcare
df = pd.read_csv("data/raw/sms_spam_raw.csv")
print(f"Încărcat: {len(df)} rânduri")


# eliminăm newline-urile din interiorul mesajelor
df["text"] = df["sms"].str.replace(r"\s+", " ", regex=True).str.strip()
df = df.drop(columns=["sms"])

# eliminăm rândurile cu text gol după curățare
df = df[df["text"].str.len() > 0].reset_index(drop=True)
print(f"După curățare: {len(df)} rânduri")

# mapare etichete
df["label_spam"]        = df["label"].astype(int)
df["label_hate_speech"] = 0
df["label_harassment"]  = 0
df["label_obscene"]     = 0
df = df.drop(columns=["label"])

# spam subtype — weak supervision
PHISHING_KEYWORDS = [
    "verify", "account", "bank", "password", "login",
    "confirm", "credentials", "secure", "update your",
    "unusual activity", "suspended", "access",
]

SCAM_KEYWORDS = [
    "you have won", "you won", "winner", "prize", "award",
    "guaranteed", "cash", "claim", "selected to receive",
    "£1000", "£2000", "£5000", "£900", "£500",
    "call to claim", "final notice", "last chance",
    "bitcoin", "crypto", "send money", "inheritance",
    "investment", "double your",
]

JUNK_KEYWORDS = [
    "free", "offer", "deal", "discount", "sale", "promo",
    "click", "txt", "call now", "limited time", "urgent",
    "ringtone", "mobile update", "camera phone", "upgrade",
    "subscription", "150p", "stop to end",
]

def assign_spam_subtype(row: pd.Series) -> str | None:
    if row["label_spam"] == 0:
        return None

    text_lower = row["text"].lower()

    if any(kw in text_lower for kw in PHISHING_KEYWORDS):
        return "phishing"
    if any(kw in text_lower for kw in SCAM_KEYWORDS):
        return "scam"
    if any(kw in text_lower for kw in JUNK_KEYWORDS):
        return "junk"

    # junk implicit pentru restul spam-urilor care nu se potrivesc cu phishing sau scam
    return "junk"


df["spam_subtype"] = df.apply(assign_spam_subtype, axis=1)

# statistici
spam_df = df[df["label_spam"] == 1]
print("\n=== Distribuție spam_subtype ===")
print(spam_df["spam_subtype"].value_counts())
print(f"Spam fără subtype: {spam_df['spam_subtype'].isna().sum()}")

# split train / val / test
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,        # reproductibilitate — același seed → aceleași split-uri
    stratify=df["label_spam"],
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label_spam"],
)

train_df = train_df.copy()
val_df   = val_df.copy()
test_df  = test_df.copy()

train_df["split"] = "train"
val_df["split"]   = "val"
test_df["split"]  = "test"

print("\n=== Dimensiuni split-uri ===")
for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
    spam_count = split["label_spam"].sum()
    print(f"{name:5}: {len(split):4} rânduri | spam: {spam_count} ({spam_count/len(split):.1%})")

# salvare
import os
os.makedirs("data/processed", exist_ok=True)

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv",     index=False)
test_df.to_csv("data/processed/test.csv",   index=False)

full_df = pd.concat([train_df, val_df, test_df])
full_df.to_csv("data/processed/full.csv", index=False)

print("\n✓ Salvat în data/processed/")
print("  train.csv / val.csv / test.csv / full.csv")

print("\n=== Preview format final (train, primele 3 rânduri) ===")
print(train_df.head(3).to_string(index=False))