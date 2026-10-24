"""Thin OpenAI-compatible HTTP client against a locally-served vLLM instance.

`model` is either the base model id (frozen debate agents) or the name of
a served LoRA adapter (the strategy generator G after at least one SFT
iteration).
"""

import time
from typing import Optional

import requests


class LLMClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def wait_until_ready(self, retries: int = 60, delay_s: float = 5.0):
        """Poll /v1/models until the server responds."""
        for attempt in range(retries):
            try:
                resp = requests.get(f"{self.base_url}/models", timeout=5.0)
                if resp.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(delay_s)
        raise RuntimeError(f"vLLM server at {self.base_url} did not become ready after {retries * delay_s:.0f}s")

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[list] = None,
    ) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = stop
        resp = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout)
        if not resp.ok:
            # Include vLLM's error body (e.g. context-length errors), which
            # raise_for_status() would drop.
            raise requests.HTTPError(f"{resp.status_code} error from {self.base_url}/chat/completions: {resp.text[:2000]}", response=resp)
        data = resp.json()
        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            print(f"[cmad] generation truncated (finish_reason=length) model={model} max_tokens={max_tokens}")
        return choice["message"]["content"]

    def load_lora_adapter(self, lora_name: str, lora_path: str) -> bool:
        """Register a LoRA adapter with the running vLLM server without a
        restart. Returns False if the server does not support runtime LoRA
        loading; use VLLMServerManager.restart_with_adapter instead."""
        try:
            resp = requests.post(
                f"{self.base_url}/load_lora_adapter",
                json={"lora_name": lora_name, "lora_path": lora_path},
                timeout=60.0,
            )
            print(f"[cmad] vllm load_lora_adapter name={lora_name} path={lora_path} status={resp.status_code}")
            return resp.status_code == 200
        except requests.RequestException as e:
            print(f"[cmad] vllm load_lora_adapter name={lora_name} path={lora_path} FAILED: {e}")
            return False

    def unload_lora_adapter(self, lora_name: str) -> bool:
        try:
            resp = requests.post(
                f"{self.base_url}/unload_lora_adapter",
                json={"lora_name": lora_name},
                timeout=30.0,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False
