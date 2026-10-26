"""在既有独立推理环境中运行固定版本的本地相关性模型。"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

if __package__:
    from mcp.qwen_embedding import QwenEmbeddingFunction, ROOT
else:
    from qwen_embedding import QwenEmbeddingFunction, ROOT

REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
MODEL_FILES = {
    "README.md": (14742, "5bba8c734f6dd3ae48317b4139317e45a7fce48fc55e15670b23a0dd15492ab6"),
    "config.json": (727, "d479c427a9ca5295218063d4f9aca4f297ab4ac27487cca7af42c84643d51ef0"),
    "model.safetensors": (1191588280, "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b"),
    "tokenizer.json": (11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    "tokenizer_config.json": (9706, "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0"),
}
INSTRUCTION = "Given a customer support question in Chinese, retrieve relevant policy or help passages that answer the question"
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def memory_peak() -> dict[str, float]:
    """读取本进程Windows内存峰值，不依赖开发实验脚本。"""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"rss_peak_mib": counters.PeakWorkingSetSize / 2**20,
            "private_peak_mib": counters.PeakPagefileUsage / 2**20}
SPEC = {"model": "Qwen/Qwen3-Reranker-0.6B", "revision": REVISION,
        "files": MODEL_FILES, "instruction": INSTRUCTION, "max_length": 1024,
        "dtype": "float32", "threads": 4, "batch_size": 1,
        "scoring": "softmax_no_yes_last_token", "prefix": PREFIX, "suffix": SUFFIX}
IDENTITY = hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()


def verify_model(path: Path) -> None:
    for name, (size, digest) in MODEL_FILES.items():
        target = path / name
        if not target.is_file() or target.stat().st_size != size:
            raise RuntimeError(f"重排模型缓存缺失或大小不符：{name}")
        with target.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                raise RuntimeError(f"重排模型文件哈希不符：{name}")


def validate_scores(scores, count: int) -> list[float]:
    if (not isinstance(scores, list) or len(scores) != count
            or any(type(x) not in (float, int) or not math.isfinite(x) or not 0 <= x <= 1 for x in scores)):
        raise ValueError("重排分数数量或数值无效")
    return scores


def passage_text(item: dict) -> str:
    return str(item.get("title") or "") + "\n" + str(item.get("content") or "")


class QwenReranker(QwenEmbeddingFunction):
    """复用向量进程的串行传输和退出清理，不共享模型权重。"""

    identity = IDENTITY

    def __init__(self, python=None, model_path=None, timeout=180):
        super().__init__(python=python, model_path=model_path or ROOT / "data/models/qwen3-reranker-0.6b", timeout=timeout)
        self.last_stats = {}

    def score(self, query: str, items: list[dict]) -> list[float]:
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise ValueError("重排问题必须为1至4000字文本")
        if (len(items) > 80 or any(not isinstance(item, dict) or not str(item.get("content") or "").strip()
                                  or len(passage_text(item)) > 12000 for item in items)):
            raise ValueError("重排候选必须有正文，最多80条且每条不超过12000字")
        with self._lock:
            if self._closed:
                raise RuntimeError("重排器已关闭")
            if not items:
                return []
            try:
                if self._process is None:
                    if not self.python.is_file():
                        raise RuntimeError("Qwen独立推理环境不存在")
                    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                               PYTHONUTF8="1", TOKENIZERS_PARALLELISM="false")
                    self._process = subprocess.Popen(
                        [str(self.python), "-u", str(Path(__file__).resolve()), str(self.model_path)],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                        text=True, encoding="utf-8", env=env, cwd=ROOT,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    self._reader = threading.Thread(target=self._read, args=(self._process.stdout,), daemon=True)
                    self._reader.start()
                    if self._receive().get("identity") != self.identity:
                        raise RuntimeError("重排进程身份不符")
                self._process.stdin.write(json.dumps({"query": query, "passages": [passage_text(i) for i in items]}, ensure_ascii=False) + "\n")
                self._process.stdin.flush()
                reply = self._receive()
                scores = validate_scores(reply.get("scores"), len(items))
                self.last_stats = reply.get("stats", {})
                return scores
            except Exception:
                self._stop()
                raise


def serve(path: Path) -> None:
    verify_model(path)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, trust_remote_code=False,
                                               dtype=torch.float32, attn_implementation="eager").eval()
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    labels = [tokenizer.convert_tokens_to_ids("no"), tokenizer.convert_tokens_to_ids("yes")]
    print(json.dumps({"identity": IDENTITY}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            scores, truncated, lengths = [], 0, []
            for passage in request["passages"]:
                text = f'<Instruct>: {INSTRUCTION}\n<Query>: {request["query"]}\n<Document>: {passage}'
                tokens = tokenizer.encode(text, add_special_tokens=False)
                limit = SPEC["max_length"] - len(prefix) - len(suffix)
                truncated += int(len(tokens) > limit)
                ids = torch.tensor([prefix + tokens[:limit] + suffix])
                lengths.append(ids.shape[1])
                with torch.inference_mode():
                    # 只计算最后一个位置的词表分数，避免保存整段词表矩阵。
                    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                   use_cache=False, logits_to_keep=1).logits[0, -1, labels]
                    scores.append(torch.softmax(logits.float(), dim=0)[1].item())
            stats = {"truncated_pairs": truncated, "token_lengths": lengths}
            if sys.platform == "win32":
                stats.update(memory_peak())
            print(json.dumps({"scores": validate_scores(scores, len(request["passages"])), "stats": stats}), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)
            return


if __name__ == "__main__":
    try:
        serve(Path(sys.argv[1]))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
        sys.exit(1)
