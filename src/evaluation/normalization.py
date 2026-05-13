import re
from decimal import Decimal, InvalidOperation
from typing import Any


MMLU_LETTERS = "ABCDEFGHIJ"
NUMERIC_PATTERN = r"-?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"


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
    numbers = re.findall(NUMERIC_PATTERN, text.replace(",", ""))
    return numbers[-1] if numbers else text.strip()


def normalize_gsm_answer(answer: Any) -> str:
    return normalize_numeric_prediction(str(answer))


def numeric_answer_variants(answer: Any) -> list[str]:
    normalized = normalize_gsm_answer(answer)
    variants = [normalized]
    try:
        value = Decimal(normalized)
    except InvalidOperation:
        return variants

    plain = format(value, "f")
    variants.append(plain)
    if "." in plain:
        stripped = plain.rstrip("0").rstrip(".")
        variants.append(stripped if stripped not in {"", "-0"} else "0")
    if value == value.to_integral_value():
        variants.append(str(int(value)))

    unique = []
    for variant in variants:
        if variant and variant not in unique:
            unique.append(variant)
    return unique


def normalize_answer(dataset: str, answer: Any) -> str:
    if dataset == "mmlu_pro":
        return normalize_mmlu_answer(answer)
    if dataset == "gsm_hard":
        return normalize_gsm_answer(answer)
    return str(answer).strip()


def normalize_prediction(dataset: str, text: str) -> str:
    if dataset == "mmlu_pro":
        return normalize_mmlu_prediction(text)
    if dataset == "gsm_hard":
        return normalize_numeric_prediction(text)
    return text.strip()
