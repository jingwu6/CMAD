"""Download MATH and write fixed-seed eval/train subsamples as JSONL, in the
same {"question": ..., "answer": ...} schema prepare_gsm8k.py writes so the
rest of the pipeline (eval.py, sft_loop.py) doesn't need to know which
dataset it's looking at.

Uses `nlile/hendrycks-MATH-benchmark` on the HF Hub, a mirror of Hendrycks
et al.'s MATH dataset (12000 train / 500 test) with a pre-extracted `answer`
field (the bare \\boxed{} content). `level` and `subject` are kept in the
output; `subject` is used for subject-aware strategy generation.

By default only levels 1-4 are used. Pass --include-level-5 for the full
1-5 range.

Usage: python3 scripts/prepare_math.py --out-dir data/ --eval-size 25 --train-pool-size 300
"""

import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--eval-size", type=int, default=None, help="Subsample size for eval; omit for full test split (500 rows)")
    parser.add_argument("--train-pool-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-dataset", default="nlile/hendrycks-MATH-benchmark")
    parser.add_argument("--include-level-5", action="store_true", help="Use the full level 1-5 range instead of the default level 1-4 subset")
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.hf_dataset)
    train_split = ds["train"]
    test_split = ds["test"]

    if not args.include_level_5:
        train_split = train_split.filter(lambda row: row.get("level") in (1, 2, 3, 4))
        test_split = test_split.filter(lambda row: row.get("level") in (1, 2, 3, 4))

    os.makedirs(args.out_dir, exist_ok=True)

    rng = random.Random(args.seed)

    train_indices = list(range(len(train_split)))
    rng.shuffle(train_indices)
    train_indices = train_indices[: args.train_pool_size]

    test_indices = list(range(len(test_split)))
    if args.eval_size is not None:
        rng2 = random.Random(args.seed)
        rng2.shuffle(test_indices)
        test_indices = test_indices[: args.eval_size]

    def write(path, split, indices):
        with open(path, "w") as f:
            for i in indices:
                row = split[i]
                f.write(json.dumps({
                    "question": row["problem"],
                    "answer": row["answer"],
                    "level": row.get("level"),
                    "subject": row.get("subject"),
                }) + "\n")

    write(os.path.join(args.out_dir, "train_pool.jsonl"), train_split, train_indices)
    write(os.path.join(args.out_dir, "eval.jsonl"), test_split, test_indices)

    print(f"[cmad] prepare_math wrote {len(train_indices)} train_pool rows, {len(test_indices)} eval rows to {args.out_dir}")


if __name__ == "__main__":
    main()
