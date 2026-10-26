"""显式获取外部公开运行资料；仓库本身不携带数据。"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


REVISION = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
BASE_URL = f"https://raw.githubusercontent.com/sierra-research/tau2-bench/{REVISION}"
ASSETS = {
    "db.json": ("data/tau2/domains/retail/db.json", "413a65160adbdb5fde0ffc0015c49b6d70250b10c18128de169b597af7766765"),
    "policy.md": ("data/tau2/domains/retail/policy.md", "2c9652afbce57d6e087768d37cda64d31c53d50b3e3225cfdb791bac66466467"),
    "LICENSE": ("LICENSE", "e67c5aa0074dfcaefd3c3a1aedb94cb539234aecd15d5a972574e3200e6252fe"),
}


def prepare(directory: Path, *, download: bool = False) -> dict[str, str]:
    """默认只检查；只有显式授权下载时联网，不覆盖已有未知内容。"""
    checked = {}
    for name, (relative, expected) in ASSETS.items():
        path = directory / name
        if path.exists():
            data = path.read_bytes()
        elif download:
            with urlopen(f"{BASE_URL}/{relative}", timeout=30) as response:
                data = response.read(8 * 1024 * 1024 + 1)
            if len(data) > 8 * 1024 * 1024:
                raise ValueError(f"外部文件超过大小上限：{name}")
        else:
            raise FileNotFoundError(f"缺少 {path}；自行提供文件，或明确使用 --download")
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f"文件哈希不符，不覆盖：{name}")
        if not path.exists():
            directory.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
        checked[name] = actual
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/retail-assets"))
    parser.add_argument("--download", action="store_true", help="明确授权从固定上游提交下载三个文件")
    args = parser.parse_args()
    for name, digest in prepare(args.directory, download=args.download).items():
        print(f"{name}: {digest}")


if __name__ == "__main__":
    main()
