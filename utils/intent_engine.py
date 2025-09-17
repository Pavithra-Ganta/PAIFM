import os
import requests
import numpy as np
from functools import lru_cache
from typing import List, Dict

class IntentEngine:
    def __init__(self):
        # ----- your original samples (unchanged) -----
        self.intent_map: Dict[str, List[str]] = {
            "greeting": ["hi", "hello", "hey","hii","good morning", "good evening", "good night"],
            "acknowledgment": ["ok", "okay", "thanks", "thank you"],
            "budget_query": ["what's my budget", "budget status","how much can i spend", "budget left"],
            "expense_query": ["expenses", "how much did i spend"],
            "expense_analysis": ["where did i spend most", "top spending category", "spending analysis", "expense breakdown", "spending habits"],
            "investment_query": ["investments", "portfolio", "sip", "mutual fund","suggest me some good stocks","where to invest"],
            "savings_advice": ["how to save", "reduce spending", "save more", "cut costs", "save money", "savings tips", "financial advice"],
            "followup": ["tell me more", "and what about"," can you explain", "more details", "can you elaborate", "what else"],
        }
        self.threshold: float = 0.6  # keep same threshold

        # ----- env config (Cloudflare Workers AI defaults) -----
        self.account_id = os.environ.get("CF_ACCOUNT_ID", "")
        default_base = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run" if self.account_id else ""
        self.base_url = os.environ.get("EMBED_BASE_URL", default_base)
        self.model = os.environ.get("EMBED_MODEL", "@cf/baai/bge-m3")
        self.api_key = os.environ.get("CF_API_TOKEN")  # required

        if not self.api_key:
            raise RuntimeError("CF_API_TOKEN is not set")
        if not self.base_url:
            raise RuntimeError("EMBED_BASE_URL is not set (or CF_ACCOUNT_ID missing)")

        # ----- precompute exemplar embeddings (batched per intent) -----
        self.intent_embs: Dict[str, List[np.ndarray]] = {
            intent: self._embed_batch(samples) for intent, samples in self.intent_map.items()
        }

    # ---------------------------
    # Embedding via API (Cloudflare / OpenAI-style fallback)
    # ---------------------------
    def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            f"{self.base_url}/{self.model}",
            json={"text": texts},      # Workers AI expects {"text": [...]}
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

    # ---- Handle multiple possible shapes ----
        vectors = None

    # 1) Workers AI common shape: {"result":{"data":[{"embedding":[...]}, ...]}}
        try:
            arr = data["result"]["data"]
            vectors = [
                np.asarray(item["embedding"] if isinstance(item, dict) else item, dtype=np.float32)
                for item in arr
            ]
        except (KeyError, TypeError):
            pass

    # 2) OpenAI-style: {"data":[{"embedding":[...]}, ...]}
        if vectors is None:
            try:
                arr = data["data"]
                vectors = [
                    np.asarray(item["embedding"] if isinstance(item, dict) else item, dtype=np.float32)
                    for item in arr
                ]
            except (KeyError, TypeError):
                pass

    # 3) Raw list of vectors: {"result":{"data":[[...],[...]]}} or {"result":[[...],[...]]}
        if vectors is None:
            try:
                arr = data["result"]["data"]
                if isinstance(arr, list) and arr and isinstance(arr[0], list):
                    vectors = [np.asarray(vec, dtype=np.float32) for vec in arr]
            except (KeyError, TypeError):
                pass

        if vectors is None:
            try:
                arr = data["result"]
                if isinstance(arr, list) and arr and isinstance(arr[0], list):
                    vectors = [np.asarray(vec, dtype=np.float32) for vec in arr]
            except (KeyError, TypeError):
                pass

    # 4) Last fallback: if top-level is a list of vectors already
        if vectors is None and isinstance(data, list) and data and isinstance(data[0], list):
            vectors = [np.asarray(vec, dtype=np.float32) for vec in data]

        if vectors is None:
        # helpful debug if shape changes again
            raise ValueError(f"Unexpected embeddings response shape: {data}")

        return vectors

        

    @lru_cache(maxsize=512)
    def _embed_text(self, text: str) -> np.ndarray:
        # cached single-text embedding
        return self._embed_batch([text])[0]

    # ---------------------------
    # Cosine similarity
    # ---------------------------
    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # ---------------------------
    # Main detection logic (unchanged)
    # ---------------------------
    def detect_intent(self, user_input: str) -> str:
        input_emb = self._embed_text(user_input)
        best_score = -1.0
        matched_intent = "unknown"

        for intent, sample_embs in self.intent_embs.items():
            if not sample_embs:
                continue
            score = max((self._cosine_sim(input_emb, e) for e in sample_embs), default=-1.0)
            if score > best_score:
                best_score = score
                matched_intent = intent

        if best_score < self.threshold:
            matched_intent = "unknown"
        print(matched_intent)
        return matched_intent
