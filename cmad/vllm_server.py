"""Manages the vLLM OpenAI-compatible server subprocess.

Newly trained LoRA adapters are made servable by restarting the server with
an updated --lora-modules list, which works regardless of whether vLLM's
runtime /v1/load_lora_adapter endpoint is available.
"""

import os
import subprocess
from typing import Dict, Optional

from cmad.llm_client import LLMClient


class VLLMServerManager:
    def __init__(
        self,
        model: str,
        port: int = 8000,
        gpu_memory_utilization: float = 0.85,
        # Debate rounds 2+ embed all M agents' previous responses as
        # history, so prompts can get long (especially on MATH).
        max_model_len: int = 16384,
        max_lora_rank: int = 32,
        cuda_visible_devices: str = "0",
        log_path: str = "vllm_server.log",
    ):
        self.model = model
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_lora_rank = max_lora_rank
        self.cuda_visible_devices = cuda_visible_devices
        self.log_path = log_path
        self.process: Optional[subprocess.Popen] = None
        self._log_file = None
        self.registered_adapters: Dict[str, str] = {}

    def _build_cmd(self):
        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--enable-lora", "--max-lora-rank", str(self.max_lora_rank),
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            "--port", str(self.port),
        ]
        if self.registered_adapters:
            cmd.append("--lora-modules")
            cmd.extend(f"{name}={path}" for name, path in self.registered_adapters.items())
        return cmd

    def start(self, retries: int = 90, delay_s: float = 5.0):
        env = os.environ.copy()
        env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        self._log_file = open(self.log_path, "a")
        print(f"[cmad] vllm_server starting: {' '.join(self._build_cmd())}")
        self.process = subprocess.Popen(self._build_cmd(), stdout=self._log_file, stderr=subprocess.STDOUT, env=env)
        LLMClient(f"http://localhost:{self.port}/v1").wait_until_ready(retries=retries, delay_s=delay_s)
        print(f"[cmad] vllm_server ready, pid={self.process.pid}, adapters={list(self.registered_adapters)}")

    def stop(self, timeout: float = 30.0):
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=timeout)
        self.process = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def restart_with_adapter(self, name: str, path: str):
        """Register a new LoRA adapter and restart the server so it becomes
        servable as model=`name`. All previously-registered adapters remain
        servable too (accumulated in self.registered_adapters, all passed
        via --lora-modules on every restart)."""
        self.registered_adapters[name] = path
        self.stop()
        self.start()

    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"
