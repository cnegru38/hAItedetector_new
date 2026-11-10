import random
import pandas as pd

# emoji specifice spam-ului grupate pe subtip

SPAM_EMOJI = {
    "junk": [
        "🎉", "🔥", "💥", "🚨", "⚡", "🎁", "🛒", "💯",
        "👉", "📣", "🆓", "🏷️", "🤩", "😍",
    ],
    "scam": [
        "💰", "🤑", "💵", "💸", "🏆", "🥇", "🎯",
        "✅", "☑️", "🔑", "💎", "👑",
    ],
    "phishing": [
        "⚠️", "🔐", "🔒", "📧", "📱", "💳", "🏦",
        "❗", "‼️", "🚫",
    ],
}

# emoji generice pentru spam fără subtype clar
GENERIC_SPAM_EMOJI = [
    "🎉", "💰", "🔥", "⚠️", "✅", "👉", "💥", "🤑", "🏆", "📣",
]


def get_emoji_pool(spam_subtype: str | None) -> list[str]:
    if spam_subtype in SPAM_EMOJI:
        # combinăm emoji specifice cu cele generice pentru varietate
        return SPAM_EMOJI[spam_subtype] + GENERIC_SPAM_EMOJI
    return GENERIC_SPAM_EMOJI


def augment_with_emoji(text: str, spam_subtype: str | None, rng: random.Random) -> str:
    pool = get_emoji_pool(spam_subtype)
    n_emoji = rng.randint(1, 3)
    chosen = rng.choices(pool, k=n_emoji)  # choices permite repetare
    emoji_str = " ".join(chosen)
    return f"{text} {emoji_str}"

# pipeline principal
def augment_dataset(
    input_path: str,
    output_path: str,
    augment_ratio: float = 0.4,
    seed: int = 42,
) -> None:

    rng = random.Random(seed)
    df = pd.read_csv(input_path)

    print(f"Încărcat: {len(df)} rânduri din {input_path}")

    # selectăm doar mesajele spam pentru augmentare
    spam_df = df[df["label_spam"] == 1].copy()
    n_to_augment = int(len(spam_df) * augment_ratio)

    # alegem aleator ce mesaje spam augmentăm
    indices_to_augment = rng.sample(list(spam_df.index), n_to_augment)
    rows_to_augment = df.loc[indices_to_augment].copy()

    print(f"Spam total      : {len(spam_df)}")
    print(f"De augmentat    : {n_to_augment} ({augment_ratio:.0%} din spam)")

    # aplicăm augmentarea
    rows_to_augment["text"] = rows_to_augment.apply(
        lambda row: augment_with_emoji(row["text"], row.get("spam_subtype"), rng),
        axis=1,
    )

    # combinăm originalul cu rândurile augmentate
    augmented_df = pd.concat([df, rows_to_augment], ignore_index=True)

    # amestecăm rândurile — important ca modelul să nu vadă
    # toate exemplele augmentate grupate la final
    augmented_df = augmented_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"Total după augmentare: {len(augmented_df)} rânduri")
    print(f"  din care spam: {augmented_df['label_spam'].sum()}")

    augmented_df.to_csv(output_path, index=False)
    print(f"✓ Salvat în {output_path}")

    # preview
    print("\n=== Exemple augmentate ===")
    sample = rows_to_augment.head(5)
    for _, row in sample.iterrows():
        print(f"  [{row.get('spam_subtype', '?')}] {row['text'][:100]}")

# rulare
if __name__ == "__main__":
    augment_dataset(
        input_path="data/processed/train_final.csv",
        output_path="data/processed/train_augmented.csv",
        augment_ratio=0.4,
        seed=42,
    )