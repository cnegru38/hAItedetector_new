from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent.parent

# încărcare surse
print("Încărcare date românești...")

ro_offense_train = pd.read_csv(BASE_DIR / "data" / "processed" / "ro_offense_train.csv")
ro_offense_test  = pd.read_csv(BASE_DIR / "data" / "processed" / "ro_offense_test.csv")
spam_ro          = pd.read_csv(BASE_DIR / "data" / "raw"       / "spam_ro.csv")

print(f"CoRoSeOf train : {len(ro_offense_train)}")
print(f"CoRoSeOf test  : {len(ro_offense_test)}")
print(f"Spam RO        : {len(spam_ro)}")

# aliniere coloane
spam_ro["label_spam"]        = 1
spam_ro["label_hate_speech"] = 0
spam_ro["label_harassment"]  = 0
spam_ro["label_obscene"]     = 0
spam_ro["hate_target"]       = None

COLUMNS = ["text", "label_spam", "label_hate_speech",
           "label_harassment", "label_obscene",
           "spam_subtype", "hate_target"]

spam_ro          = spam_ro[[c for c in COLUMNS if c in spam_ro.columns]]
ro_offense_train = ro_offense_train[[c for c in COLUMNS if c in ro_offense_train.columns]]
ro_offense_test  = ro_offense_test[[c for c in COLUMNS if c in ro_offense_test.columns]]

# split spam RO
spam_train, spam_val = train_test_split(
    spam_ro, test_size=0.15, random_state=42
)

spam_train["split"] = "train"
spam_val["split"]   = "val"

# split CoRoSeOf 
# stratificăm pe o coloană combinată
ro_offense_train["_strat"] = (
    ro_offense_train["label_hate_speech"] * 1 +
    ro_offense_train["label_harassment"]  * 2 +
    ro_offense_train["label_obscene"]     * 4
)

ro_train, ro_val = train_test_split(
    ro_offense_train,
    test_size=0.15,
    random_state=42,
    stratify=ro_offense_train["_strat"]
)

ro_train = ro_train.drop(columns=["_strat"])
ro_val   = ro_val.drop(columns=["_strat"])

ro_train["split"] = "train"
ro_val["split"]   = "val"
ro_offense_test["split"] = "test"

print(f"\n=== Distribuție split-uri RO ===")
for name, df in [("train (offense)", ro_train), ("val (offense)", ro_val),
                 ("test (offense)",  ro_offense_test),
                 ("train (spam)",    spam_train), ("val (spam)", spam_val)]:
    hs  = df["label_hate_speech"].sum()
    har = df["label_harassment"].sum()
    obs = df["label_obscene"].sum()
    sp  = df["label_spam"].sum()
    print(f"  {name:20}: {len(df):4} | spam:{sp:3} | hate:{hs:3} | haras:{har:3} | obsc:{obs:3}")

# integrare în dataset
train_existing = pd.read_csv(BASE_DIR / "data" / "processed" / "train_final.csv")
val_existing   = pd.read_csv(BASE_DIR / "data" / "processed" / "val_final.csv")
test_existing  = pd.read_csv(BASE_DIR / "data" / "processed" / "test_final.csv")


def align_and_concat(existing: pd.DataFrame, *new_dfs) -> pd.DataFrame:
    """Aliniază coloanele și concatenează toate DataFrame-urile."""
    parts = [existing]
    for new_df in new_dfs:
        aligned = new_df.copy()
        for col in existing.columns:
            if col not in aligned.columns:
                aligned[col] = 0 if "label" in col else None
        aligned = aligned[[c for c in existing.columns if c in aligned.columns]]
        parts.append(aligned)
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


train_final = align_and_concat(train_existing, ro_train, spam_train)
val_final   = align_and_concat(val_existing,   ro_val,   spam_val)
test_final  = align_and_concat(test_existing,  ro_offense_test)

train_final.to_csv(BASE_DIR / "data" / "processed" / "train_final.csv", index=False)
val_final.to_csv(BASE_DIR / "data" / "processed"   / "val_final.csv",   index=False)
test_final.to_csv(BASE_DIR / "data" / "processed"  / "test_final.csv",  index=False)

print(f"\n=== Dataset final actualizat ===")
for name, df in [("train", train_final), ("val", val_final), ("test", test_final)]:
    sp  = df["label_spam"].sum()
    hs  = df["label_hate_speech"].sum()
    har = df["label_harassment"].sum()
    obs = df["label_obscene"].sum()
    nor = ((df[["label_spam","label_hate_speech",
                "label_harassment","label_obscene"]].sum(axis=1)) == 0).sum()
    print(f"{name:5}: {len(df):6} | spam:{sp:4} | hate:{hs:.0f} | haras:{har:.0f} | obsc:{obs:.0f} | normal:{nor}")

print("\n✓ train_final.csv / val_final.csv / test_final.csv actualizate cu date RO")