"""通过独立 Python 环境运行固定 Qwen 向量模型，隔离 Chroma 的依赖。"""
from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
INSTRUCTION = "Given a customer support question in Chinese, retrieve relevant policy or help passages that answer the question"
MODEL_FILES = {
    "config.json": (727, "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9"),
    "model.safetensors": (1191586416, "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"),
    "tokenizer.json": (11423705, "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"),
    "tokenizer_config.json": (9706, "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0"),
    "README.md": (17237, "c34d9b7e5a267ad3fdd13227a253686bc90844ff4744a2a6a86c7c905e3d06f3"),
}
SPEC = {"model": "Qwen/Qwen3-Embedding-0.6B", "revision": REVISION,
        "files": MODEL_FILES, "dimension": 1024, "pooling": "last_token_left_padding",
        "normalization": "l2", "max_length": 512, "query_instruction": INSTRUCTION,
        "dtype": "float32", "threads": 4, "distance": "l2"}
IDENTITY = hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()


def verify_model(path: Path) -> None:
    """运行时只读本地固定文件，不联网或执行模型仓库代码。"""
    for name, (size, expected) in MODEL_FILES.items():
        target = path / name
        if not target.is_file() or target.stat().st_size != size:
            raise RuntimeError(f"Qwen 缓存缺失或大小不符：{name}；请显式准备模型")
        with target.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != expected:
            raise RuntimeError(f"Qwen 文件哈希不符：{name}")


def validate_vectors(vectors: list, count: int) -> list:
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ValueError("Qwen 返回向量数量不符")
    for vector in vectors:
        if (not isinstance(vector, list) or len(vector) != 1024
                or any(type(x) not in (float, int) or not math.isfinite(x) for x in vector)
                or abs(sum(x * x for x in vector) - 1) > 1e-4):
            raise ValueError("Qwen 返回无效向量或未归一化")
    return vectors


class QwenEmbeddingFunction:
    """单运行时复用一个串行编码进程；超时或退出后显式报错。"""

    identity = IDENTITY
    collection_suffix = "_qwen_" + IDENTITY[:12]

    def __init__(self, python: Path | None = None, model_path: Path | None = None,
                 timeout: float = 180):
        self.python = python or ROOT / "data/embedding-env/Scripts/python.exe"
        self.model_path = model_path or ROOT / "data/models/qwen3-embedding-0.6b"
        self.timeout = timeout
        self._lock = threading.Lock()
        self._process = None
        self._reader = None
        self._replies = queue.Queue()
        self._closed = False
        atexit.register(self.close)

    def __call__(self, input):
        import numpy as np
        # Chroma 0.5 的自动文档编码接口要求每行是 NumPy 向量。
        return np.asarray(self._encode(list(input), False), dtype=np.float32)

    def encode_query(self, query: str):
        return self._encode([query], True)

    def _read(self, stream):
        try:
            for line in stream:
                self._replies.put(json.loads(line))
        except Exception:
            self._replies.put({"error": "Qwen 输出协议无效"})
        finally:
            self._replies.put({"error": "Qwen 编码进程已退出"})

    def _receive(self):
        try:
            reply = self._replies.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"等待 Qwen 编码响应超时（{self.timeout} 秒）") from exc
        if not isinstance(reply, dict) or "error" in reply:
            raise RuntimeError(str(reply))
        return reply

    def _encode(self, texts: list[str], is_query: bool):
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Qwen 输入必须为非空文本")
        with self._lock:
            if self._closed:
                raise RuntimeError("Qwen 编码器已关闭")
            if not texts:
                return []
            try:
                if self._process is None:
                    if not self.python.is_file():
                        raise RuntimeError("Qwen 独立环境缺失，请按 README 安装推理依赖")
                    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                               PYTHONUTF8="1", TOKENIZERS_PARALLELISM="false")
                    self._process = subprocess.Popen(
                        [str(self.python), "-u", str(Path(__file__).resolve()), str(self.model_path)],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                        text=True, encoding="utf-8", env=env, cwd=ROOT,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    self._reader = threading.Thread(target=self._read, args=(self._process.stdout,), daemon=True)
                    self._reader.start()
                    if self._receive().get("identity") != self.identity:
                        raise RuntimeError("Qwen 编码进程身份不符")
                self._process.stdin.write(json.dumps({"texts": texts, "query": is_query}, ensure_ascii=False) + "\n")
                self._process.stdin.flush()
                return validate_vectors(self._receive().get("vectors"), len(texts))
            except Exception as exc:
                self._stop()
                raise RuntimeError(f"Qwen 编码失败，未回退到其他模型：{exc}") from exc

    def _stop(self):
        self._closed = True
        process = self._process
        if process is not None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if self._reader is not None:
                self._reader.join(timeout=5)
            process.stdin.close()
            process.stdout.close()
        atexit.unregister(self.close)

    def close(self):
        with self._lock:
            self._stop()


def serve(model_path: Path) -> None:
    verify_model(model_path)
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True,
                                              trust_remote_code=False, padding_side="left")
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True,
                                     trust_remote_code=False, dtype=torch.float32,
                                     attn_implementation="eager").eval()
    print(json.dumps({"identity": IDENTITY}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            vectors = []
            # batch=1 与固定对照一致，限制多文档导入的瞬时内存。
            for text in request["texts"]:
                if request["query"]:
                    text = f"Instruct: {INSTRUCTION}\nQuery: {text}"
                batch = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
                with torch.inference_mode():
                    hidden = model(**batch).last_hidden_state[:, -1]
                    vectors.append(torch.nn.functional.normalize(hidden, p=2, dim=1)[0].cpu().tolist())
            print(json.dumps({"vectors": validate_vectors(vectors, len(request["texts"]))}), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)
            return


if __name__ == "__main__":
    try:
        serve(Path(sys.argv[1]))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
        sys.exit(1)
