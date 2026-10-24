"""Full-run driver: baselines (cot/vanilla_mad/dmad) + the two CMAD variants
(with/without diversity selection), each evaluated after every SFT
iteration. Writes one JSON per condition to --results-dir.

Needs two GPUs: one serves vLLM (--vllm-gpu), the other runs LoRA
fine-tuning in this process (--train-gpu).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmad.config import DATASET_CONFIGS, DebateConfig, LoRAConfig, SelectionConfig, SFTLoopConfig


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_result(results_dir, name, result):
    path = os.path.join(results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump({name: result}, f, indent=2)
    print(f"[cmad] wrote {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-iterations", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--adapter-dir", default="run_adapters")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--debate-m", type=int, default=3)
    parser.add_argument("--debate-n", type=int, default=2)
    parser.add_argument("--vllm-gpu", default="0", help="CUDA device id(s) for the vLLM server")
    parser.add_argument("--train-gpu", default="1", help="CUDA device id(s) for LoRA fine-tuning")
    parser.add_argument("--skip-baselines", action="store_true",
                         help="Skip cot/vanilla_mad/dmad (they do not depend on the SFT loop).")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), default="gsm8k",
                         help="Prompt wording + answer extractor/scorer. Must match the "
                              "prepare_{gsm8k,math}.py script that produced the data files.")
    parser.add_argument("--lora-lr", type=float, default=None, help="Override LoRAConfig.lr (default 1e-4)")
    parser.add_argument("--lora-epochs", type=int, default=None, help="Override LoRAConfig.num_train_epochs (default 3)")
    parser.add_argument("--no-alignment", action="store_true", help="Disable the alignment agent (Eq 8)")
    args = parser.parse_args()

    # Must be set before torch initializes CUDA (LoRA training runs in this
    # process); the vLLM subprocess gets its own device via VLLMServerManager.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.train_gpu

    from cmad.eval import run_condition
    from cmad.llm_client import LLMClient
    from cmad.sft_loop import run_two_variant_sft_loops
    from cmad.vllm_server import VLLMServerManager

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.adapter_dir, exist_ok=True)

    dataset_cfg = DATASET_CONFIGS[args.dataset]()

    # subject is None for GSM8K; on MATH it enables subject-aware strategy generation.
    train_questions = [{"question": row["question"], "subject": row.get("subject")} for row in load_jsonl(args.train_file)]
    eval_questions = load_jsonl(args.eval_file)
    print(f"[cmad] loaded {len(train_questions)} train questions, {len(eval_questions)} eval questions, dataset={args.dataset}")

    server = VLLMServerManager(model=args.base_model, cuda_visible_devices=args.vllm_gpu)
    server.start()
    client = LLMClient(server.base_url())

    debate_cfg = DebateConfig(M=args.debate_m, N=args.debate_n, model=args.base_model)

    if not args.skip_baselines:
        for name in ("cot", "vanilla_mad", "dmad"):
            result = run_condition(name, client, eval_questions, model=args.base_model,
                                    debate_M=args.debate_m, debate_N=args.debate_n, concurrency=args.concurrency,
                                    dataset_cfg=dataset_cfg)
            save_result(args.results_dir, name, result)

    selection_cfg_with_diversity = SelectionConfig(apply_diversity=True)
    selection_cfg_no_diversity = SelectionConfig(apply_diversity=False)
    lora_cfg = LoRAConfig()
    if args.lora_lr is not None:
        lora_cfg.lr = args.lora_lr
    if args.lora_epochs is not None:
        lora_cfg.num_train_epochs = args.lora_epochs
    loop_cfg = SFTLoopConfig(L=args.num_iterations)
    print(f"[cmad] lora lr={lora_cfg.lr} epochs={lora_cfg.num_train_epochs} use_alignment={not args.no_alignment}")

    cmad_adapters, no_div_adapters = run_two_variant_sft_loops(
        client, train_questions, args.base_model, debate_cfg,
        selection_cfg_with_diversity, selection_cfg_no_diversity,
        lora_cfg, loop_cfg, args.adapter_dir, server,
        use_alignment=not args.no_alignment, concurrency=args.concurrency, dataset_cfg=dataset_cfg,
    )

    for variant_name, adapters in (("cmad", cmad_adapters), ("cmad_no_diversity", no_div_adapters)):
        for i, adapter_name in enumerate(adapters, start=1):
            result = run_condition(
                "cmad", client, eval_questions, model=args.base_model,
                debate_M=args.debate_m, debate_N=args.debate_n,
                generator_model=adapter_name, concurrency=args.concurrency, dataset_cfg=dataset_cfg,
            )
            save_result(args.results_dir, f"{variant_name}_iter{i}", result)

    server.stop()
    print("[cmad] run_cmad_full done")


if __name__ == "__main__":
    main()
