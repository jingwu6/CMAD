"""Download GSM8K and write fixed-seed eval/train subsamples as JSONL.

Usage: python3 scripts/prepare_gsm8k.py --out-dir data/ --eval-size 25 --train-pool-size 300
"""

import argparse
import json
import os
import random
import re


def _extract_gt_answer(answer_field: str) -> str:
    # GSM8K's own answer field ends with "#### <number>".
    match = re.search(r"####\s*(-?[0-9,]+(?:\.[0-9]+)?)", answer_field)
    assert match, f"could not find GSM8K ground-truth answer in: {answer_field!r}"
    return match.group(1).replace(",", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--eval-size", type=int, default=None, help="Subsample size for eval; omit for full test split")
    parser.add_argument("--train-pool-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")
    train_split = ds["train"]
    test_split = ds["test"]

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
                f.write(json.dumps({"question": row["question"], "answer": _extract_gt_answer(row["answer"])}) + "\n")

    write(os.path.join(args.out_dir, "train_pool.jsonl"), train_split, train_indices)
    write(os.path.join(args.out_dir, "eval.jsonl"), test_split, test_indices)

    print(f"[cmad] prepare_gsm8k wrote {len(train_indices)} train_pool rows, {len(test_indices)} eval rows to {args.out_dir}")


if __name__ == "__main__":
    main()
