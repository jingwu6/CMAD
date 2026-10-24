"""CPU-only embedding wrapper for the diversity-selection metric (selection.py).

Embeds the concatenation of strategy + final-answer text: two differently
worded strategies can still produce homogeneous reasoning, which is the
collapse mode diversity selection is meant to catch.
"""

import threading
import time
from typing import List

import numpy as np

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    # embed_candidates is called from worker threads (sft_loop.py); the lock
    # ensures the model is constructed exactly once, since concurrent PyTorch
    # module construction is not thread-safe.
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            last_exc = None
            for attempt in range(3):
                try:
                    _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu",
                                                  model_kwargs={"low_cpu_mem_usage": False})
                    break
                except NotImplementedError as e:
                    last_exc = e
                    print(f"[cmad] embeddings model load attempt {attempt + 1}/3 failed: {e}")
                    time.sleep(2.0)
            else:
                raise last_exc
    return _model


def embed_candidates(strategies: List[str], final_answers_raw: List[str]) -> np.ndarray:
    """Return an (N, D) array of L2-normalized embeddings, one per
    (strategy, final_answer) pair."""
    texts = [f"{s}\n\n{a}" for s, a in zip(strategies, final_answers_raw)]
    model = _get_model()
    embeddings = np.asarray(model.encode(texts, convert_to_numpy=True))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return embeddings / norms
