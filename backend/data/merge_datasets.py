import os
import pandas as pd

# coloane așteptate

COLUMNS = [
    "text",
    "label_spam",
    "label_hate_speech",
    "label_harassment",
    "label_obscene",
    "spam_subtype",
    "hate_target",
    "split",
]

# încărcare
print("Încărcare dataset-uri...")

spam_train = pd.read_csv("data/processed/train.csv")
spam_val   = pd.read_csv("data/processed/val.csv")
spam_test  = pd.read_csv("data/processed/test.csv")

hate_train = pd.read_csv("data/processed/hatexplain_train.csv")
hate_val   = pd.read_csv("data/processed/hatexplain_val.csv")
hate_test  = pd.read_csv("data/processed/hatexplain_test.csv")

print(f"Spam     — train: {len(spam_train)} | val: {len(spam_val)} | test: {len(spam_test)}")
print(f"HateXplain — train: {len(hate_train)} | val: {len(hate_val)} | test: {len(hate_test)}")

# aliniere coloane
def align_columns(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    df = df.copy()

    # adăugăm coloanele lipsă
    if "label_obscene" not in df.columns:
        df["label_obscene"] = 0
    if "hate_target" not in df.columns:
        df["hate_target"] = None
    if "spam_subtype" not in df.columns:
        df["spam_subtype"] = None

    # setăm split-ul corect
    df["split"] = split_name

    # eliminăm coloane extra care nu fac parte din format
    extra_cols = [c for c in df.columns if c not in COLUMNS]
    if extra_cols:
        df = df.drop(columns=extra_cols)

    # reordonăm coloanele
    df = df[COLUMNS]

    # normalizăm tipurile
    for col in ["label_spam", "label_hate_speech", "label_harassment", "label_obscene"]:
        df[col] = df[col].fillna(0).astype(int)

    return df


spam_train = align_columns(spam_train, "train")
spam_val   = align_columns(spam_val,   "val")
spam_test  = align_columns(spam_test,  "test")
hate_train = align_columns(hate_train, "train")
hate_val   = align_columns(hate_val,   "val")
hate_test  = align_columns(hate_test,  "test")

# combinare și amestecare

train_final = pd.concat([spam_train, hate_train], ignore_index=True)
val_final   = pd.concat([spam_val,   hate_val],   ignore_index=True)
test_final  = pd.concat([spam_test,  hate_test],  ignore_index=True)

train_final = train_final.sample(frac=1, random_state=42).reset_index(drop=True)
val_final   = val_final.sample(frac=1,   random_state=42).reset_index(drop=True)
test_final  = test_final.sample(frac=1,  random_state=42).reset_index(drop=True)

# statistici finale
print("\n=== Statistici dataset final ===")
for name, df in [("train", train_final), ("val", val_final), ("test", test_final)]:
    total      = len(df)
    spam       = df["label_spam"].sum()
    hate       = df["label_hate_speech"].sum()
    harassment = df["label_harassment"].sum()
    normal     = ((df[["label_spam", "label_hate_speech", "label_harassment"]].sum(axis=1)) == 0).sum()

    print(f"\n{name.upper()} — {total} rânduri")
    print(f"  spam       : {spam:5} ({spam/total:.1%})")
    print(f"  hate_speech: {hate:5} ({hate/total:.1%})")
    print(f"  harassment : {harassment:5} ({harassment/total:.1%})")
    print(f"  normal     : {normal:5} ({normal/total:.1%})")

# verificare: niciun rând cu text gol
empty = train_final["text"].isna().sum() + (train_final["text"] == "").sum()
print(f"\nRânduri cu text gol în train: {empty}")

# salvare
os.makedirs("data/processed", exist_ok=True)

train_final.to_csv("data/processed/train_final.csv", index=False)
val_final.to_csv("data/processed/val_final.csv",     index=False)
test_final.to_csv("data/processed/test_final.csv",   index=False)

print("\n✓ Salvat în data/processed/")
print("  train_final.csv / val_final.csv / test_final.csv")

full_df = pd.concat([train_final, val_final, test_final], ignore_index=True)
full_df.to_csv("data/processed/full.csv", index=False)
print("  full.csv")

print("\n=== Preview train_final (primele 5 rânduri) ===")
print(train_final.head().to_string(index=False))