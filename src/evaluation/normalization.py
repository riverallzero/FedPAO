import re
from typing import Any


MMLU_LETTERS = "ABCDEFGHIJ"


def normalize_mmlu_answer(answer: Any) -> str:
    if isinstance(answer, int):
        return MMLU_LETTERS[answer] if 0 <= answer < len(MMLU_LETTERS) else str(answer)

    text = str(answer).strip().upper()
    if text.isdigit():
        index = int(text)
        return MMLU_LETTERS[index] if 0 <= index < len(MMLU_LETTERS) else text
    return text


def normalize_mmlu_prediction(text: str) -> str:
    normalized = text.strip().upper()
    patterns = [
        r"ANSWER\s*(?:IS|:)?\s*\(?([A-J])\)?",
        r"OPTION\s*\(?([A-J])\)?",
        r"^\s*\(?([A-J])\)?\s*$",
        r"\b([A-J])\b\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(1)
    return text.strip()


def normalize_numeric_prediction(text: str) -> str:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else text.strip()


def normalize_answer(dataset: str, answer: Any) -> str:
    if dataset == "mmlu_pro":
        return normalize_mmlu_answer(answer)
    if dataset == "gsm_hard":
        return str(answer).strip()
    return str(answer).strip()


def normalize_prediction(dataset: str, text: str) -> str:
    if dataset == "mmlu_pro":
        return normalize_mmlu_prediction(text)
    if dataset == "gsm_hard":
        return normalize_numeric_prediction(text)
    return text.strip()
