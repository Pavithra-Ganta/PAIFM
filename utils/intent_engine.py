import os
import requests
import numpy as np
from functools import lru_cache
from typing import List, Dict

class IntentEngine:
    def __init__(self):
        self.intent_map: Dict[str, List[str]] = {
            "greeting": ["hi", "hello", "hey","hii","good morning", "good evening", "good night"],
            "acknowledgment": ["ok", "okay", "thanks", "thank you"],
            "budget_query": ["what's my budget", "budget status","how much can i spend", "budget left"],
            "expense_query": ["expenses", "how much did i spend"],
            "expense_analysis": ["where did i spend most", "top spending category", "spending analysis", "expense breakdown", "spending habits"],
            "investment_query": ["investments", "portfolio", "sip", "mutual fund","suggest me some good stocks","where to invest"],
            "savings_advice": ["how to save", "reduce spending", "save more", "cut costs", "save money", "savings tips", "financial advice"],
            "followup": ["tell me more", "and what about","can you explain", "more details", "can you elaborate", "what else"],
            "balance": ["what's my balance", "account balance", "check balance"],
            "transfer": ["send money", "transfer cash", "pay friend"],
        }
        self.threshold: float = 0.6  

        # Cloudflare setup
        self.account_id = os.environ.get("CF_ACCOUNT_ID", "")
        default_base = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run" if self.account_id else ""
        self.base_url = os.environ.get("EMBED_BASE_URL", default_base)
        self.model = os.environ.get("EMBED_MODEL", "@cf/baai/bge-m3")
        self.api_key = os.environ.get("CF_API_TOKEN")

        if not self.api_key:
            raise RuntimeError("CF_API_TOKEN is not set")
        if not self.base_url:
            raise RuntimeError("EMBED_BASE_URL is not set (or CF_ACCOUNT_ID missing)")

        self.intent_embs: Dict[str, List[np.ndarray]] = {}
        # ❗ Instead of precomputing here (which causes Render timeout), delay it
        # Call warmup() after deploy to fill intent_embs
        print("⚠️ Intent embeddings not yet loaded. Call /chatbot/warmup to initialize.")

    def warmup(self):
        """Precompute intent embeddings after deploy (to avoid cold start crash)."""
        print("🔄 Warming up intent embeddings...")
        self.intent_embs = {
            intent: self._embed_batch(samples) for intent, samples in self.intent_map.items()
        }
        print("✅ Intent embeddings ready!")

    def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = requests.post(
            f"{self.base_url}/{self.model}",
            json={"text": texts},
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 429:
            print("⚠️ Rate limited by Cloudflare. Consider retry/backoff.")
        resp.raise_for_status()
        data = resp.json()

        # flexible parsing
        vectors = None
        try:
            arr = data["result"]["data"]
            vectors = [np.asarray(item.get("embedding", item), dtype=np.float32) for item in arr]
        except Exception:
            try:
                arr = data.get("data", [])
                vectors = [np.asarray(item.get("embedding", item), dtype=np.float32) for item in arr]
            except Exception:
                raise ValueError(f"Unexpected embeddings response: {data}")
        return vectors

    @lru_cache(maxsize=512)
    def _embed_text(self, text: str) -> np.ndarray:
        return self._embed_batch([text])[0]

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def detect_intent(self, user_input: str) -> str:
        if not self.intent_embs:
            print("⚠️ Warning: intent embeddings not preloaded. Call /warmup first.")
            return "unknown"

        input_emb = self._embed_text(user_input)
        best_score, matched_intent = -1.0, "unknown"

        for intent, sample_embs in self.intent_embs.items():
            if not sample_embs:
                continue
            score = max((self._cosine_sim(input_emb, e) for e in sample_embs), default=-1.0)
            if score > best_score:
                best_score, matched_intent = score, intent

        return matched_intent if best_score >= self.threshold else "unknown"
