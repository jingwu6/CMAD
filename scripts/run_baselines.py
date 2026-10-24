"""CLI entrypoint for running baseline conditions (cot, vanilla_mad, dmad).

Usage examples:
  python3 scripts/run_baselines.py --eval-file data/eval.jsonl --conditions cot vanilla_mad dmad \
      --model Qwen/Qwen2.5-7B-Instruct --out results_baselines.json

  python3 scripts/run_baselines.py --eval-file data/eval.jsonl --conditions cmad \
      --model Qwen/Qwen2.5-7B-Instruct --generator-model cmad_iter1 --out results_cmad_iter1.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmad.config import DATASET_CONFIGS
from cmad.eval import run_condition
from cmad.llm_client import LLMClient


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--generator-model", default=None)
    parser.add_argument("--debate-m", type=int, default=3)
    parser.add_argument("--debate-n", type=int, default=2)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), default="gsm8k")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset_cfg = DATASET_CONFIGS[args.dataset]()
    questions = load_jsonl(args.eval_file)
    client = LLMClient(base_url=args.base_url)
    client.wait_until_ready()

    all_results = {}
    for condition in args.conditions:
        result = run_condition(
            condition,
            client,
            questions,
            model=args.model,
            debate_M=args.debate_m,
            debate_N=args.debate_n,
            generator_model=args.generator_model,
            dataset_cfg=dataset_cfg,
        )
        all_results[condition] = result

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[cmad] run_baselines wrote results for {list(all_results.keys())} to {args.out}")


if __name__ == "__main__":
    main()
