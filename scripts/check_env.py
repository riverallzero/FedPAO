import importlib.util
import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


print("Python:", sys.version)
print("Working directory:", os.getcwd())

packages = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "evaluate",
    "cma",
    "flwr",
    "openai",
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "yaml",
    "pydantic",
]

for package in packages:
    print(f"{package:15s}:", "OK" if has_package(package) else "MISSING")

try:
    import torch

    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device count:", torch.cuda.device_count())
        print("cuda device 0:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch check failed:", repr(exc))
