from pathlib import Path
import pandas as pd
from datasets import load_dataset

BASE_DIR = Path(__file__).parent.parent

N_OFFENSIVE = 2000  # exemple obscene EN
N_NORMAL    = 1000  # exemple normale EN din același dataset

SLURS = [
    'nigga', 'nigger', 'nazi', 'chink', 'spic', 'kike',
    'faggot', 'fag', 'dyke', 'tranny', 'retard', 'cripple',
    'wetback', 'beaner', 'gook', 'towelhead', 'sandnigger',
]
WIKIPEDIA_TERMS = [
    'Wikipedia', 'wiki', 'RFC', 'edit', 'article', 'vandal',
    'arbitration', 'CamelCase', 'template', 'disambiguation',
]

slur_pattern = '|'.join(SLURS)
wiki_pattern = '|'.join(WIKIPEDIA_TERMS)

# încărcare și filtrare
print("Descărcare tarekziade/profanity...")
ds = load_dataset('tarekziade/profanity', split='train')
df = ds.to_pandas()
print(f"Total: {len(df)}")

df['word_count'] = df['text'].str.split().str.len()
df['text']       = df['text'].astype(str).str.strip()
df               = df[df['text'].str.len() >= 3]

# filtrare offensive
offensive_df = df[
    (df['is_offensive'] == 1) &
    (df['word_count'] <= 15) &
    (~df['text'].str.contains(slur_pattern, case=False, na=False)) &
    (~df['text'].str.contains(wiki_pattern, case=False, na=False))
].copy()

# filtrare normale
normal_df = df[
    (df['is_offensive'] == 0) &
    (df['word_count'] <= 15) &
    (~df['text'].str.contains(wiki_pattern, case=False, na=False))
].copy()

print(f"Offensive disponibile: {len(offensive_df)}")
print(f"Normale disponibile  : {len(normal_df)}")

# selectare subset
offensive_sample = offensive_df.sample(
    min(N_OFFENSIVE, len(offensive_df)), random_state=42
)
normal_sample = normal_df.sample(
    min(N_NORMAL, len(normal_df)), random_state=42
)

print(f"\nSelectat offensive: {len(offensive_sample)}")
print(f"Selectat normale  : {len(normal_sample)}")

print("\n=== Exemple offensive ===")
for text in offensive_sample['text'].head(5).tolist():
    print(f"  {text[:80]}")

# format unificat
offensive_final = pd.DataFrame({
    "text":               offensive_sample['text'].values,
    "label_spam":         0,
    "label_hate_speech":  0,
    "label_harassment":   0,
    "label_obscene":      1,
    "spam_subtype":       None,
    "hate_target":        None,
})

normal_final = pd.DataFrame({
    "text":               normal_sample['text'].values,
    "label_spam":         0,
    "label_hate_speech":  0,
    "label_harassment":   0,
    "label_obscene":      0,
    "spam_subtype":       None,
    "hate_target":        None,
})

combined = pd.concat([offensive_final, normal_final], ignore_index=True)

# deduplicare
existing = pd.read_csv(BASE_DIR / "data" / "processed" / "full.csv")
existing_texts = set(existing["text"].str.lower().str.strip())

before = len(combined)
combined = combined[
    ~combined["text"].str.lower().str.strip().isin(existing_texts)
].copy()
print(f"\nDupă deduplicare: {len(combined)} (eliminat {before - len(combined)} duplicate)")

# split și integrare în dataset
combined_shuffled = combined.sample(frac=1, random_state=42).reset_index(drop=True)
n       = len(combined_shuffled)
n_test  = max(1, int(n * 0.15))
n_val   = max(1, int(n * 0.15))
n_train = n - n_test - n_val

prof_train = combined_shuffled.iloc[:n_train].copy()
prof_val   = combined_shuffled.iloc[n_train:n_train + n_val].copy()
prof_test  = combined_shuffled.iloc[n_train + n_val:].copy()

prof_train["split"] = "train"
prof_val["split"]   = "val"
prof_test["split"]  = "test"

print(f"\n=== Distribuție split-uri ===")
for name, split in [("train", prof_train), ("val", prof_val), ("test", prof_test)]:
    obs = split["label_obscene"].sum()
    print(f"{name:5}: {len(split):4} | obscene: {obs}")

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

train_final = align_and_concat(train_existing, prof_train)
val_final   = align_and_concat(val_existing,   prof_val)
test_final  = align_and_concat(test_existing,  prof_test)

train_final.to_csv(BASE_DIR / "data" / "processed" / "train_final.csv", index=False)
val_final.to_csv(BASE_DIR / "data" / "processed"   / "val_final.csv",   index=False)
test_final.to_csv(BASE_DIR / "data" / "processed"  / "test_final.csv",  index=False)

print(f"\n=== Dataset final actualizat ===")
for name, df_f in [("train", train_final), ("val", val_final), ("test", test_final)]:
    obs  = df_f["label_obscene"].sum()
    spam = df_f["label_spam"].sum()
    hs   = df_f["label_hate_speech"].sum()
    print(f"{name:5}: {len(df_f):6} | spam:{spam:4} | hate:{hs:.0f} | obscene:{obs:.0f}")

print("\n✓ train_final.csv / val_final.csv / test_final.csv actualizate")