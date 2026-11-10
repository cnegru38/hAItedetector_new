
import json
import os
from collections import Counter

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

# descărcare
URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"

print("Descărcare HateXplain...")
response = requests.get(URL)
response.raise_for_status()
data = response.json()
print(f"Descărcat: {len(data)} posturi")

# procesare

def majority_label(annotators: list) -> str:
    labels = [a["label"] for a in annotators]
    count = Counter(labels)
    top_label, top_count = count.most_common(1)[0]
    # acceptăm doar dacă cel puțin 2 din 3 sunt de acord
    if top_count >= 2:
        return top_label
    return "normal"  # fallback


def majority_target(annotators: list) -> str | None:
    all_targets = []
    for a in annotators:
        all_targets.extend(a.get("target", []))

    # eliminăm 'None' și valori goale
    all_targets = [t for t in all_targets if t and t.lower() != "none"]
    if not all_targets:
        return None

    most_common = Counter(all_targets).most_common(1)[0][0]
    return map_target(most_common)


# mapare ținte HateXplain
_TARGET_MAP: dict[str, str] = {
    "African":     "race",
    "Arab":        "ethnicity",
    "Asian":       "race",
    "Caucasian":   "race",
    "Hispanic":    "ethnicity",
    "Islam":       "religion",
    "Jewish":      "religion",
    "Homosexual":  "orientation",
    "Women":       "gender",
    "Refugee":     "ethnicity",
    "Migrants":    "ethnicity",
    "Men":         "gender",
    "Christian":   "religion",
    "Hindu":       "religion",
}


def map_target(target: str) -> str:
    return _TARGET_MAP.get(target, "other")

# construire DataFrame
rows = []
skipped = 0

for post_id, post in data.items():
    tokens = post.get("post_tokens", [])
    if not tokens:
        skipped += 1
        continue

    # reconstruim textul din tokens
    text = " ".join(tokens)

    label = majority_label(post["annotators"])
    target = majority_target(post["annotators"])

    rows.append({
        "text":               text,
        "label_spam":         0,   # HateXplain nu conține spam
        "label_hate_speech":  1 if label == "hatespeech" else 0,
        "label_harassment":   1 if label == "offensive" else 0,
        "label_obscene":      0,   # nu avem adnotări obscene în acest dataset
        "spam_subtype":       None,
        "hate_target":        target if label == "hatespeech" else None,
    })

df = pd.DataFrame(rows)
print(f"\nProcessat: {len(df)} rânduri (skipped: {skipped})")


# statistici
print("\n=== Distribuție etichete ===")
print(f"hate_speech : {df['label_hate_speech'].sum()} ({df['label_hate_speech'].mean():.1%})")
print(f"harassment  : {df['label_harassment'].sum()} ({df['label_harassment'].mean():.1%})")
print(f"normal      : {(df['label_hate_speech'] + df['label_harassment'] == 0).sum()}")

print("\n=== Distribuție hate_target ===")
print(df[df["hate_target"].notna()]["hate_target"].value_counts())

# split train / val / test

# creăm o coloană combinată pentru stratificare multi-label
# 0 = normal, 1 = hate_speech, 2 = harassment
df["_strat"] = (
    df["label_hate_speech"] * 1 + df["label_harassment"] * 2
)

train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["_strat"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["_strat"]
)

# ștergem coloana temporară de stratificare
for split in [train_df, val_df, test_df]:
    split.drop(columns=["_strat"], inplace=True)

train_df["split"] = "train"
val_df["split"]   = "val"
test_df["split"]  = "test"

print("\n=== Dimensiuni split-uri ===")
for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
    hs = split["label_hate_speech"].sum()
    har = split["label_harassment"].sum()
    print(f"{name:5}: {len(split):5} rânduri | hate: {hs} ({hs/len(split):.1%}) | harassment: {har} ({har/len(split):.1%})")

# salvare
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# salvăm și raw pentru referință
with open("data/raw/hatexplain_raw.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

train_df.to_csv("data/processed/hatexplain_train.csv", index=False)
val_df.to_csv("data/processed/hatexplain_val.csv",     index=False)
test_df.to_csv("data/processed/hatexplain_test.csv",   index=False)

print("\n✓ Salvat în data/processed/")
print("  hatexplain_train.csv / hatexplain_val.csv / hatexplain_test.csv")

print("\n=== Preview (3 exemple hate speech) ===")
for _, row in train_df[train_df["label_hate_speech"] == 1].head(3).iterrows():
    print(f"  [target: {row['hate_target']}] {row['text'][:100]}")

print("\n=== Preview (3 exemple harassment) ===")
for _, row in train_df[train_df["label_harassment"] == 1].head(3).iterrows():
    print(f"  {row['text'][:100]}")