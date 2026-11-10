from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent.parent

# încărcare și agregare pe comment_id
print("Descărcare measuring-hate-speech...")
ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
df = ds.to_pandas()
print(f"Total adnotări: {len(df)}")
print(f"Comentarii unice: {df['comment_id'].nunique()}")

# agregare pe comment_id
agg_df = df.groupby("comment_id").agg(
    text              = ("text", "first"),
    hate_speech_score = ("hate_speech_score", "mean"),
    insult            = ("insult", "mean"),
    hatespeech        = ("hatespeech", "mean"),
    target_race       = ("target_race", "max"),
    target_religion   = ("target_religion", "max"),
    target_gender     = ("target_gender", "max"),
    target_sexuality  = ("target_sexuality", "max"),
    target_origin     = ("target_origin", "max"),
).reset_index()

print(f"\nDupă agregare: {len(agg_df)} comentarii unice")
print(f"\nDistribuție hate_speech_score:")
print(f"  > 0.5  (hate speech) : {(agg_df['hate_speech_score'] > 0.5).sum()}")
print(f"  -1 to 0.5 (neutru)   : {((agg_df['hate_speech_score'] >= -1.0) & (agg_df['hate_speech_score'] <= 0.5)).sum()}")
print(f"  < -1.0 (counter)     : {(agg_df['hate_speech_score'] < -1.0).sum()}")

# mapare țintă hate speech
def map_hate_target(row) -> str | None:
    if row["target_race"]:       return "race"
    if row["target_religion"]:   return "religion"
    if row["target_gender"]:     return "gender"
    if row["target_sexuality"]:  return "orientation"
    if row["target_origin"]:     return "ethnicity"
    return "other"

# selecție stratificată
N_HATE       = 2000   # exemple hate speech clare
N_HARASSMENT = 1000   # exemple harassment (insulte fără hate clar)
N_NORMAL     = 1500   # exemple normale clare (counter-speech și neutru)

# hate speech clar
hate_df = agg_df[agg_df["hate_speech_score"] > 0.5].copy()
hate_df = hate_df.sample(min(N_HATE, len(hate_df)), random_state=42)
hate_df["label_hate_speech"] = 1
hate_df["label_harassment"]  = 0
hate_df["hate_target"]       = hate_df.apply(map_hate_target, axis=1)
print(f"\nHate speech selectat: {len(hate_df)}")

# harassment = insulte fără hate speech clar
harassment_df = agg_df[
    (agg_df["insult"] > 1.0) &
    (agg_df["hate_speech_score"] < 0.5) &
    (agg_df["hate_speech_score"] > -0.5)
].copy()
harassment_df = harassment_df.sample(min(N_HARASSMENT, len(harassment_df)), random_state=42)
harassment_df["label_hate_speech"] = 0
harassment_df["label_harassment"]  = 1
harassment_df["hate_target"]       = None
print(f"Harassment selectat: {len(harassment_df)}")

# normale clare = scor foarte negativ = contrar hate speech sau neutru clar
normal_df = agg_df[agg_df["hate_speech_score"] < -1.0].copy()
normal_df = normal_df.sample(min(N_NORMAL, len(normal_df)), random_state=42)
normal_df["label_hate_speech"] = 0
normal_df["label_harassment"]  = 0
normal_df["hate_target"]       = None
print(f"Normal selectat: {len(normal_df)}")

# combinare și format unificat
combined = pd.concat([hate_df, harassment_df, normal_df], ignore_index=True)
combined["label_spam"]    = 0
combined["label_obscene"] = 0
combined["spam_subtype"]  = None

combined = combined[[
    "text", "label_spam", "label_hate_speech",
    "label_harassment", "label_obscene",
    "spam_subtype", "hate_target"
]]

# curățare text
combined["text"] = combined["text"].astype(str).str.strip()
combined = combined[combined["text"].str.len() >= 3].copy()

print(f"\n=== Distribuție finală subset ===")
print(f"hate_speech : {combined['label_hate_speech'].sum()}")
print(f"harassment  : {combined['label_harassment'].sum()}")
print(f"normal      : {(combined['label_hate_speech'] + combined['label_harassment'] == 0).sum()}")

# deduplicare față de dataset-ul existent
existing = pd.read_csv(BASE_DIR / "data" / "processed" / "full.csv")
existing_texts = set(existing["text"].str.lower().str.strip())

before = len(combined)
combined = combined[
    ~combined["text"].str.lower().str.strip().isin(existing_texts)
].copy()
print(f"\nDupă deduplicare: {len(combined)} (eliminat {before - len(combined)} duplicate)")

# split și integrare
# stratificăm pe o coloană combinată
combined["_strat"] = combined["label_hate_speech"] * 1 + combined["label_harassment"] * 2

combined_shuffled = combined.sample(frac=1, random_state=42).reset_index(drop=True)
n       = len(combined_shuffled)
n_test  = max(1, int(n * 0.15))
n_val   = max(1, int(n * 0.15))
n_train = n - n_test - n_val

mhs_train = combined_shuffled.iloc[:n_train].copy()
mhs_val   = combined_shuffled.iloc[n_train:n_train + n_val].copy()
mhs_test  = combined_shuffled.iloc[n_train + n_val:].copy()

for split_df, name in [(mhs_train, "train"), (mhs_val, "val"), (mhs_test, "test")]:
    split_df["split"] = name
    split_df.drop(columns=["_strat"], inplace=True, errors="ignore")

print(f"\n=== Distribuție split-uri ===")
for name, split in [("train", mhs_train), ("val", mhs_val), ("test", mhs_test)]:
    hs  = split["label_hate_speech"].sum()
    har = split["label_harassment"].sum()
    print(f"{name:5}: {len(split):4} | hate: {hs} | harassment: {har}")

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


train_final = align_and_concat(train_existing, mhs_train)
val_final   = align_and_concat(val_existing,   mhs_val)
test_final  = align_and_concat(test_existing,  mhs_test)

train_final.to_csv(BASE_DIR / "data" / "processed" / "train_final.csv", index=False)
val_final.to_csv(BASE_DIR / "data" / "processed" / "val_final.csv",     index=False)
test_final.to_csv(BASE_DIR / "data" / "processed" / "test_final.csv",   index=False)

print(f"\n=== Dataset final actualizat ===")
for name, split in [("train", train_final), ("val", val_final), ("test", test_final)]:
    hs  = split["label_hate_speech"].sum()
    har = split["label_harassment"].sum()
    spam = split["label_spam"].sum()
    print(f"{name:5}: {len(split):5} | spam: {spam} | hate: {hs} | harassment: {har}")

print("\n✓ train_final.csv / val_final.csv / test_final.csv actualizate")