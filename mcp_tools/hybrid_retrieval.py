"""小型冻结知识库的 BM25 与等权 RRF，不引入额外检索服务。"""
from collections import Counter
import math
import re
import unicodedata

BM25_K1 = 1.2
BM25_B = 0.75
RRF_CONSTANT = 60
CANDIDATE_K = 20


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"\bwireless\s+earbuds\b", "无线耳机", text)
    text = re.sub(r"\bbluetooth\b", "蓝牙", text)
    tokens = []
    for part in re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+(?:[._-][a-z0-9]+)*", text):
        if re.fullmatch(r"[\u3400-\u9fff]+", part) and len(part) > 1:
            tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


class BM25Index:
    def __init__(self, items: list[dict]):
        self.items = {item["evidence_id"]: item for item in items}
        if len(self.items) != len(items):
            raise ValueError("BM25 证据ID重复")
        self.terms = {}
        for key, item in self.items.items():
            text = " ".join(str(item.get(field) or "") for field in
                            ("title", "content", "evidence_id", "product_id", "step_id"))
            self.terms[key] = Counter(tokenize(text))
        self.lengths = {key: sum(terms.values()) for key, terms in self.terms.items()}
        self.average_length = sum(self.lengths.values()) / max(len(items), 1) or 1
        self.df = Counter(term for terms in self.terms.values() for term in terms)

    def search(self, query: str, top_k: int, allowed: set[str]) -> list[dict]:
        scores = []
        query_terms = set(tokenize(query))
        for key in sorted(allowed):
            terms = self.terms[key]
            norm = BM25_K1 * (1 - BM25_B + BM25_B * self.lengths[key] / self.average_length)
            score = sum(math.log(1 + (len(self.items) - self.df[term] + 0.5) / (self.df[term] + 0.5))
                        * terms[term] * (BM25_K1 + 1) / (terms[term] + norm)
                        for term in query_terms if terms[term])
            if score > 0:
                scores.append((key, score))
        scores.sort(key=lambda pair: (-pair[1], pair[0]))
        return [{**self.items[key], "score": score, "bm25_score": score,
                 "retrieval_mode": "bm25"} for key, score in scores[:top_k]]


def reciprocal_rank_fusion(dense: list[dict], lexical: list[dict], top_k: int) -> list[dict]:
    items, scores, ranks = {}, Counter(), {}
    for channel, rows in (("dense", dense), ("bm25", lexical)):
        seen = set()
        for rank, item in enumerate(rows, 1):
            key = item["evidence_id"]
            if key in seen:
                continue
            seen.add(key)
            items.setdefault(key, item)
            scores[key] += 1 / (RRF_CONSTANT + rank)
            ranks.setdefault(key, {})[channel] = rank
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    return [{**items[key], "score": scores[key], "rrf_score": scores[key],
             "channel_ranks": ranks[key], "retrieval_mode": "hybrid"} for key in ordered]
