from pathlib import Path
from collections import Counter

import pandas as pd
from datasets import load_dataset

BASE_DIR = Path(__file__).parent.parent

# încărcare
print("Descărcare CoRoSeOf...")
ds_train = load_dataset("readerbench/ro-offense", split="train")
ds_test  = load_dataset("readerbench/ro-offense", split="test")

df_train = ds_train.to_pandas()
df_test  = ds_test.to_pandas()

print(f"Train: {len(df_train)} | Test: {len(df_test)}")
print(f"\nDistribuție etichete train:")
print(Counter(df_train["label"].tolist()))


# mapare etichete
LABEL_MAP = {
    "ABUSE":     {"label_hate_speech": 1, "label_harassment": 0, "label_obscene": 0},
    "INSULT":    {"label_hate_speech": 0, "label_harassment": 1, "label_obscene": 0},
    "PROFANITY": {"label_hate_speech": 0, "label_harassment": 0, "label_obscene": 1},
    "OTHER":     {"label_hate_speech": 0, "label_harassment": 0, "label_obscene": 0},
}

def process_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        label_info = LABEL_MAP.get(row["label"], LABEL_MAP["OTHER"])
        rows.append({
            "text":               row["text"],
            "label_spam":         0,
            "label_hate_speech":  label_info["label_hate_speech"],
            "label_harassment":   label_info["label_harassment"],
            "label_obscene":      label_info["label_obscene"],
            "spam_subtype":       None,
            "hate_target":        None,
        })
    return pd.DataFrame(rows)

processed_train = process_df(df_train)
processed_test  = process_df(df_test)

print(f"\n=== Distribuție finală train ===")
print(f"hate_speech : {processed_train['label_hate_speech'].sum()}")
print(f"harassment  : {processed_train['label_harassment'].sum()}")
print(f"obscene     : {processed_train['label_obscene'].sum()}")
print(f"normal      : {((processed_train[['label_hate_speech','label_harassment','label_obscene']].sum(axis=1)) == 0).sum()}")


print("\n=== Exemple per categorie ===")
for col in ["label_hate_speech", "label_harassment", "label_obscene"]:
    sample = processed_train[processed_train[col] == 1]["text"].iloc[0]
    print(f"[{col}] {sample[:80]}")

# salvare
out_dir = BASE_DIR / "data" / "processed"
out_dir.mkdir(exist_ok=True)

processed_train.to_csv(out_dir / "ro_offense_train.csv", index=False)
processed_test.to_csv(out_dir  / "ro_offense_test.csv",  index=False)

print(f"\n✓ Salvat în data/processed/")
print(f"  ro_offense_train.csv ({len(processed_train)} rânduri)")
print(f"  ro_offense_test.csv  ({len(processed_test)} rânduri)")