from decimal import Decimal, InvalidOperation

from .normalization import normalize_answer, normalize_prediction


def exact_match(dataset: str, prediction: str, ground_truth: str) -> bool:
    if dataset == "gsm_hard":
        try:
            return Decimal(prediction) == Decimal(ground_truth)
        except InvalidOperation:
            return False
    return prediction == ground_truth


def score_prediction(dataset: str, response: str, answer: str) -> dict:
    prediction = normalize_prediction(dataset, response)
    ground_truth = normalize_answer(dataset, answer)
    correct = exact_match(dataset, prediction, ground_truth)
    return {
        "prediction": prediction,
        "ground_truth": ground_truth,
        "correct": correct,
        "accuracy": 1.0 if correct else 0.0,
    }
