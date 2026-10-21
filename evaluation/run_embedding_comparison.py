"""U04 中文向量对照：独立编码进程、共用 Chroma 协议，不调用生成模型。"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
WEIGHT_SHA256 = "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd"
MODEL_FILES = {
    "config.json": (727, "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9"),
    "model.safetensors": (1191586416, WEIGHT_SHA256),
    "tokenizer.json": (11423705, "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a"),
    "tokenizer_config.json": (9706, "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0"),
    "README.md": (17237, "c34d9b7e5a267ad3fdd13227a253686bc90844ff4744a2a6a86c7c905e3d06f3"),
}
INSTRUCTION = "Given a customer support question in Chinese, retrieve relevant policy or help passages that answer the question"
INPUTS = (
    "evaluation/data/retail_sandbox/knowledge.jsonl",
    "evaluation/data/retail_support/knowledge.jsonl",
    "evaluation/data/retail/retrieval.jsonl",
    "evaluation/data/embedding_comparison/support_queries.jsonl",
)
ROUNDS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_inputs() -> tuple[list[dict], list[dict]]:
    batches = [[json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines() if line.strip()] for name in INPUTS]
    passages, queries = batches[0] + batches[1], batches[2] + batches[3]
    known = {p["evidence_id"] for p in passages}
    if len(known) != len(passages) or len({q["query_id"] for q in queries}) != len(queries):
        raise ValueError("知识或查询 ID 重复")
    for query in queries:
        if query["language"] != "zh" or query["split"] not in ("dev", "test", "support_smoke"):
            raise ValueError("本次只运行冻结的中文划分")
        relevant = {key for key, grade in query["qrels"].items() if grade >= 2}
        groups = query["required_evidence_groups"]
        if not set(query["qrels"]) <= known:
            raise ValueError("相关性标注引用未知证据")
        if query["answerability"] == "answerable":
            if not relevant or not groups or any(not group or not set(group) <= relevant for group in groups):
                raise ValueError("可答题缺少有效的直接证据组")
        elif relevant or groups:
            raise ValueError("工具事实与无答案题不得伪造相关证据")
    return passages, queries


def query_text(text: str) -> str:
    return f"Instruct: {INSTRUCTION}\nQuery: {text}"


def verify_qwen(path: Path) -> dict:
    manifest = json.loads((path / "model-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("model") != "Qwen/Qwen3-Embedding-0.6B" or manifest.get("revision") != REVISION:
        raise ValueError("Qwen 模型身份或提交与冻结配置不符")
    if set(manifest.get("files") or {}) != set(MODEL_FILES):
        raise ValueError("本地模型文件清单不完整")
    for name, item in manifest["files"].items():
        size, digest = MODEL_FILES[name]
        if item["sha256"] != digest or item["size"] != size or (path / name).stat().st_size != size or sha256(path / name) != digest:
            raise ValueError(f"模型文件哈希不符：{name}")
    if manifest["files"]["model.safetensors"]["sha256"] != WEIGHT_SHA256:
        raise ValueError("权重不是冻结的官方文件")
    return manifest


def prepare_model(path: Path) -> dict:
    """只有显式准备命令可联网；校验缓存，拒绝覆盖未知或损坏文件。"""
    path.mkdir(parents=True, exist_ok=True)
    if (path / "model-manifest.json").exists():
        return verify_qwen(path)
    files = {}
    for name, (size, digest) in MODEL_FILES.items():
        target = path / name
        url = f"https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/{REVISION}/{name}"
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=path, suffix=".part", delete=False) as temporary:
                temp = Path(temporary.name)
            try:
                with urllib.request.urlopen(url, timeout=60) as response, temp.open("wb") as handle:
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > size:
                            raise ValueError(f"模型下载超过固定大小：{name}")
                        handle.write(chunk)
                if temp.stat().st_size != size or sha256(temp) != digest:
                    raise ValueError(f"模型下载校验失败：{name}")
                if target.exists():
                    raise FileExistsError(f"模型文件已存在，拒绝覆盖：{target}")
                temp.rename(target)
            finally:
                temp.unlink(missing_ok=True)
        if target.stat().st_size != size or sha256(target) != digest:
            raise ValueError(f"现有文件不匹配，未覆盖：{target}")
        files[name] = {"sha256": digest, "size": size, "source": url}
    manifest = {"model": "Qwen/Qwen3-Embedding-0.6B", "revision": REVISION,
                "license": "Apache-2.0", "files": files}
    write_json(path / "model-manifest.json", manifest)
    return verify_qwen(path)


def memory_peak() -> dict[str, float]:
    """读取 Windows 累计峰值，包含载入峰值而非仅推理后的瞬时内存。"""
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage", "PrivateUsage",
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


def check_vectors(documents: Any, queries: Any, document_count: int, query_count: int, dimension: int) -> None:
    import numpy as np

    if documents.shape != (document_count, dimension) or queries.shape != (ROUNDS, query_count, dimension):
        raise ValueError("向量形状或维度不符，拒绝混用索引")
    for array in (documents, queries):
        if not np.isfinite(array).all() or not np.allclose(np.linalg.norm(array, axis=-1), 1, atol=1e-4):
            raise ValueError("向量非有限值或未正确归一化")


def worker(kind: str, output: Path, model_path: Path) -> None:
    import numpy as np
    from importlib.metadata import version

    passages, queries = load_inputs()
    started = time.perf_counter()
    if kind == "minilm":
        from mcp.knowledge_base import _EMBEDDING_REQUIRED_FILES, _ProjectEmbeddingFunction, _embedding_cache_ready

        if not _embedding_cache_ready():
            raise RuntimeError("MiniLM 缓存不完整，不允许隐式下载")
        encoder = _ProjectEmbeddingFunction(preferred_providers=["CPUExecutionProvider"])
        folder = Path(encoder.DOWNLOAD_PATH) / "onnx"
        model_files = {name: sha256(folder / name) for name in _EMBEDDING_REQUIRED_FILES}
        spec = {"model": "all-MiniLM-L6-v2", "files": model_files, "dimension": 384,
                "pooling": "attention_mask_mean", "normalization": "l2", "max_length": 256,
                "query_instruction": "", "dtype": "float32", "threads": "original_onnx_default"}
        encoder(["模型预热"])

        def encode(text: str, is_query: bool):
            return np.asarray(encoder([text])[0], dtype=np.float32)

        def length(text: str, is_query: bool):
            encoder.tokenizer.no_truncation()
            try:
                return sum(encoder.tokenizer.encode(text).attention_mask)
            finally:
                encoder.tokenizer.enable_truncation(max_length=256)

        packages = {name: version(name) for name in ("numpy", "onnxruntime", "tokenizers", "chromadb")}
        disk_bytes = sum((folder / name).stat().st_size for name in model_files)
    else:
        manifest = verify_qwen(model_path)
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False, padding_side="left")
        model = AutoModel.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False,
                                          dtype=torch.float32, attn_implementation="eager").eval()
        spec = {"model": manifest["model"], "revision": REVISION, "files": manifest["files"],
                "dimension": 1024, "pooling": "last_token_left_padding", "normalization": "l2",
                "max_length": 512, "query_instruction": INSTRUCTION, "dtype": "float32", "threads": 4}

        def encode(text: str, is_query: bool):
            batch = tokenizer([query_text(text) if is_query else text], padding=True, truncation=True,
                              max_length=512, return_tensors="pt")
            with torch.inference_mode():
                hidden = model(**batch).last_hidden_state[:, -1]
                return torch.nn.functional.normalize(hidden, p=2, dim=1)[0].cpu().numpy().copy()

        def length(text: str, is_query: bool):
            return len(tokenizer(query_text(text) if is_query else text, truncation=False)["input_ids"])

        packages = {name: version(name) for name in ("torch", "transformers", "tokenizers", "numpy")}
        disk_bytes = sum(item["size"] for item in manifest["files"].values())
        encode("模型预热", False)
    load_ms = (time.perf_counter() - started) * 1000
    lengths = {"documents": [length(p["content"], False) for p in passages],
               "queries": [length(q["query"], True) for q in queries]}
    started = time.perf_counter()
    documents = np.stack([encode(p["content"], False) for p in passages])
    document_ms = (time.perf_counter() - started) * 1000
    query_vectors, durations = [], []
    for round_index in range(ROUNDS):
        vectors, elapsed = [], []
        for q in queries:
            started = time.perf_counter()
            vectors.append(encode(q["query"], True))
            elapsed.append((time.perf_counter() - started) * 1000)
        query_vectors.append(vectors)
        durations.append(elapsed)
        print(f"{kind}: encoding round {round_index + 1}/{ROUNDS} complete", flush=True)
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    check_vectors(documents, query_vectors, len(passages), len(queries), spec["dimension"])
    np.savez_compressed(output / f"{kind}.npz", documents=documents, queries=query_vectors)
    write_json(output / f"{kind}_encoding.json", {
        "spec": spec, "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        "packages": packages, "python": sys.version, "device": "cpu", "batch_size": 1,
        "load_and_warmup_ms": load_ms, "document_encoding_ms": document_ms,
        "query_encoding_ms": durations, "token_lengths": lengths,
        "truncated_documents": [p["evidence_id"] for p, n in zip(passages, lengths["documents"]) if n > spec["max_length"]],
        "truncated_queries": [q["query_id"] for q, n in zip(queries, lengths["queries"]) if n > spec["max_length"]],
        "model_disk_bytes": disk_bytes, **memory_peak(),
    })


def score_rows(queries: list[dict], rankings: list[list[str]], known: set[str]) -> dict:
    from evaluation.retail_metrics import compute_retrieval_metrics

    if len(rankings) != len(queries):
        raise ValueError("查询与排序数量不一致，不能缩小评分分母")
    return {split: compute_retrieval_metrics([
        {"query_id": q["query_id"], "split": split, "answerability": q["answerability"],
         "gold_qrels": q["qrels"], "required_groups": q["required_evidence_groups"], "ranking": ranking}
        for q, ranking in zip(queries, rankings) if q["split"] == split
    ], known_ids=known) for split in ("dev", "test", "support_smoke")}


def recommend(baseline: dict, candidate: dict) -> dict:
    a, b = baseline["metrics"], candidate["metrics"]
    checks = {
        "dev_all_evidence_gain_at_least_0_10": b["dev"]["all_evidence_hit@5"] - a["dev"]["all_evidence_hit@5"] >= 0.10 - 1e-9,
        "dev_recall_improves": b["dev"]["recall@5"] > a["dev"]["recall@5"],
        "test_all_evidence_no_regression": b["test"]["all_evidence_hit@5"] >= a["test"]["all_evidence_hit@5"],
        "test_recall_no_regression": b["test"]["recall@5"] >= a["test"]["recall@5"],
        "support_no_regression": b["support_smoke"]["all_evidence_hit@5"] >= a["support_smoke"]["all_evidence_hit@5"],
        "cpu_warm_p95_within_1000_ms": candidate["warm_latency"]["p95_ms"] <= 1000,
        "private_peak_within_3_gib": candidate["encoding"]["private_peak_mib"] <= 3072,
        "protocol_clean": all(m["protocol_clean"] for m in list(a.values()) + list(b.values())),
    }
    return {"checks": checks, "candidate_passed": all(checks.values()),
            "recommendation": "QWEN_SEPARATE_RUNTIME_CANDIDATE" if all(checks.values()) else "RETAIN_MINILM",
            "runtime_default": "minilm", "deployment": "NOT_INTEGRATED_TOKENIZER_CONFLICT"}


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import chromadb
    from evaluation.retail_metrics import summarize_latency
    from mcp.knowledge_base import KnowledgeBase

    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录非空，拒绝覆盖既有证据")
    output.mkdir(parents=True, exist_ok=True)
    passages, queries = load_inputs()
    frozen = {name: sha256(ROOT / name) for name in (
        *INPUTS, "evaluation/run_embedding_comparison.py", "evaluation/embedding_requirements.lock",
        "evaluation/embedding_requirements.in", "mcp/knowledge_base.py", "evaluation/retail_metrics.py",
    )}
    write_json(output / "frozen.json", {"inputs": frozen, "platform": platform.platform(), "processor": platform.processor(),
                                       "rounds": ROUNDS, "acceptance": {"dev_gain": 0.10, "cpu_p95_ms": 1000, "private_peak_mib": 3072}})
    write_json(output / "run.json", {"status": "RUNNING", "paid_api_calls": 0})
    results = {}
    try:
        client = chromadb.HttpClient(host=args.chroma_host, port=args.chroma_port,
                                    settings=chromadb.Settings(anonymized_telemetry=False))
        client.heartbeat()
        original_collections = sorted(c.name for c in client.list_collections())
        for kind, python in (("minilm", sys.executable), ("qwen", str(Path(args.qwen_python).resolve()))):
            command = [python, "-X", "utf8", "-m", "evaluation.run_embedding_comparison", "--worker", kind,
                       "--output", str(output), "--model-path", str(Path(args.model_path).resolve())]
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"}
            subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=1800)
            encoding = json.loads((output / f"{kind}_encoding.json").read_text(encoding="utf-8"))
            with np.load(output / f"{kind}.npz", allow_pickle=False) as vectors:
                documents, query_vectors = vectors["documents"], vectors["queries"]
            check_vectors(documents, query_vectors, len(passages), len(queries), encoding["spec"]["dimension"])
            name = f"u04_{output.name[-16:].lower().replace('-', '_')}_{kind}_{encoding['spec_sha256'][:8]}"
            collection = client.create_collection(name=name, embedding_function=None, metadata={
                "hnsw:space": "l2", "embedding_spec_sha256": encoding["spec_sha256"],
                "corpus_sha256": hashlib.sha256(json.dumps({name: frozen[name] for name in INPUTS[:2]}, sort_keys=True).encode()).hexdigest(),
            })
            try:
                started = time.perf_counter()
                collection.add(ids=[p["evidence_id"] for p in passages], documents=[p["content"] for p in passages],
                               metadatas=[KnowledgeBase._evidence_metadata(p) for p in passages], embeddings=documents.tolist())
                index_ms = (time.perf_counter() - started) * 1000
                rows, all_durations, warm_durations, stable = [], [], [], True
                for round_index in range(ROUNDS):
                    rankings, times = [], []
                    for i, query in enumerate(queries):
                        started = time.perf_counter()
                        found = collection.query(query_embeddings=[query_vectors[round_index, i].tolist()], n_results=5)
                        query_ms = (time.perf_counter() - started) * 1000
                        rankings.append(found["ids"][0])
                        times.append(query_ms + encoding["query_encoding_ms"][round_index][i])
                    if round_index == 0:
                        rows = rankings
                    else:
                        stable = stable and rankings == rows
                        warm_durations.extend(times)
                    all_durations.append(times)
                result = {"collection": name, "index_ms": index_ms, "encoding": encoding,
                          "metrics": score_rows(queries, rows, {p["evidence_id"] for p in passages}),
                          "query_latency_ms": all_durations, "first_round_latency": summarize_latency(all_durations[0]),
                          "warm_latency": summarize_latency(warm_durations), "ranking_stable": stable}
                write_json(output / f"{kind}_results.json", result)
                results[kind] = result
            finally:
                client.delete_collection(name)
        if any(sha256(ROOT / name) != digest for name, digest in frozen.items()):
            raise RuntimeError("运行期间代码或评测输入变化，不能形成冻结对照")
        known = {p["evidence_id"] for p in passages}
        summary = {
            "status": "EVALUATION_COMPLETE", "models": results, "selection": recommend(results["minilm"], results["qwen"]),
            "full_context": {"passages": len(passages), "characters": sum(len(p["content"]) for p in passages),
                             "required_group_coverage_upper_bound": all(all(set(g) & known for g in q["required_evidence_groups"]) for q in queries if q["answerability"] == "answerable"),
                             "answer_quality": "NOT_RUN", "note": "仅证据覆盖上界，不是生成质量对照"},
            "paid_api_calls": 0, "cost_cny": 0,
        }
        after = sorted(c.name for c in client.list_collections())
        write_json(output / "collections.json", {"before": original_collections, "after": after})
        if after != original_collections:
            raise RuntimeError("集合列表未恢复，需人工核查")
        write_json(output / "summary.json", summary)
        write_json(output / "run.json", {"status": "EVALUATION_COMPLETE", "paid_api_calls": 0})
    except Exception as error:
        write_json(output / "run.json", {"status": "RUN_ERROR", "error": str(error), "traceback": traceback.format_exc(), "paid_api_calls": 0})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--prepare-model", action="store_true", help="显式下载或校验固定模型，并生成本地来源清单")
    parser.add_argument("--model-path", default=str(ROOT / "data/models/qwen3-embedding-0.6b"))
    parser.add_argument("--qwen-python", default=str(ROOT / "data/embedding-env/Scripts/python.exe"))
    parser.add_argument("--chroma-host", default="127.0.0.1")
    parser.add_argument("--chroma-port", type=int, default=8002)
    parser.add_argument("--worker", choices=("minilm", "qwen"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepare_model:
        manifest = prepare_model(Path(args.model_path))
        print(json.dumps({"model": manifest["model"], "revision": manifest["revision"], "verified_files": len(manifest["files"])}, ensure_ascii=False))
        return
    if not args.output:
        parser.error("评测必须指定新的 --output 目录")
    if args.worker:
        worker(args.worker, Path(args.output), Path(args.model_path))
    else:
        run(args)


if __name__ == "__main__":
    main()
