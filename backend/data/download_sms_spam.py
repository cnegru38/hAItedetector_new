from datasets import load_dataset
import pandas as pd

# descărcare
print("Descărcare dataset ucirvine/sms_spam ...")
ds = load_dataset("ucirvine/sms_spam", split="train")

# conversie la pandas
df = ds.to_pandas()

# inspecție de bază

print("\n=== Primele 5 rânduri ===")
print(df.head())

print("\n=== Coloane disponibile ===")
print(df.columns.tolist())

print("\n=== Distribuție etichete ===")
print(df["label"].value_counts())
print(f"\nTotal mesaje : {len(df)}")
print(f"Spam         : {(df['label'] == 1).sum()} ({(df['label'] == 1).mean():.1%})")
print(f"Ham (normal) : {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.1%})")

print("\n=== Lungime medie mesaje ===")
df["length"] = df["sms"].str.len()
print(f"Spam  — medie: {df[df['label'] == 1]['length'].mean():.0f} caractere")
print(f"Ham   — medie: {df[df['label'] == 0]['length'].mean():.0f} caractere")

print("\n=== 3 exemple spam ===")
for _, row in df[df["label"] == 1].head(3).iterrows():
    print(f"  [{row['length']} chars] {row['sms'][:120]}")

print("\n=== 3 exemple ham ===")
for _, row in df[df["label"] == 0].head(3).iterrows():
    print(f"  [{row['length']} chars] {row['sms'][:120]}")


# salvare
import os
os.makedirs("data/raw", exist_ok=True)

df.to_csv("data/raw/sms_spam_raw.csv", index=False)
print("\n✓ Salvat în data/raw/sms_spam_raw.csv")