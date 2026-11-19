"""
Step 1 - Merge and clean the five Part-1 labeled datasets.

Output: data/part1_clean.csv with columns
    text    cleaned text (space-separated tokens)
    label   0 = Negative, 1 = Neutral, 2 = Positive
    domain  one of the five project domains
    split   "train" or "test" (fixed here, reused by every later script)

Run:  python Code/data_preprocessor.py
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from text_utils import detect_domain, normalize_text_az, to_class_label

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "part1_labeled"
OUT_CSV = ROOT / "data" / "part1_clean.csv"
SEED = 42
TEST_SIZE = 0.2

# (file name, text column, label column, label scheme, project domain)
FILES = [
    ("labeled-sentiment.xlsx",      "text", "sentiment", "tri",    "Public Services"),
    ("merged_dataset_CSV__1_.xlsx", "text", "labels",    "binary", "Retail & Lifestyle"),
    ("test__1_.xlsx",               "text", "label",     "binary", "Finance & Business"),
    ("train-00000-of-00001.xlsx",   "text", "labels",    "tri",    "Technology & Digital Services"),
    ("train__3_.xlsx",              "text", "label",     "binary", "Social Life & Entertainment"),
]


def load_one(fname, text_col, label_col, scheme, domain):
    df = pd.read_excel(IN_DIR / fname)
    df = df[[text_col, label_col]]
    df = df.rename(columns={text_col: "raw_text", label_col: "raw_label"})
    df["label"] = df["raw_label"].apply(lambda v: to_class_label(v, scheme))
    df["domain"] = domain
    print(f"{fname}: {len(df)} line")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["raw_text"]).copy()
    df["text"] = df["raw_text"].astype(str).apply(lambda s: normalize_text_az(s, domain=detect_domain(s)))
    df = df.dropna(subset=["label"])
    df = df[df["text"] != ""]
    df["label"] = df["label"].astype(int)
    df = df[["text", "label", "domain"]]
    conflicts = df.groupby("text")["label"].nunique()
    print("conflicts:", (conflicts > 1).sum())
    df = df.drop_duplicates(subset=["text"])
    return df


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    strat = df["domain"] + "_" + df["label"].astype(str)
    train_idx, test_idx = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=SEED, stratify=strat)
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    print(f"train: {len(train_idx)}, test: {len(test_idx)}")
    return df


def main() -> None:
    frames = []
    for row in FILES:
       frames.append(load_one(*row))
    df = pd.concat(frames, ignore_index=True)
    df = clean(df)
    df = add_split(df)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(df)} lines to {OUT_CSV}")
    print(len(df))
    print(pd.crosstab(df["domain"], df["label"]))
    print(df["split"].value_counts(normalize=True))
    
    



if __name__ == "__main__":
    main()
