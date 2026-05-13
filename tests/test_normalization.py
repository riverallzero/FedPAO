from src.evaluation.normalization import (
    normalize_mmlu_answer,
    normalize_prediction,
)
from src.evaluation.metrics import exact_match


def test_mmlu_answer_normalization():
    assert normalize_mmlu_answer("a") == "A"
    assert normalize_mmlu_answer(1) == "B"
    assert normalize_mmlu_answer("2") == "C"


def test_mmlu_prediction_normalization():
    assert normalize_prediction("mmlu_pro", "The answer is A.") == "A"
    assert normalize_prediction("mmlu_pro", "option B") == "B"
    assert normalize_prediction("mmlu_pro", "I think the answer is C") == "C"
    assert normalize_prediction("mmlu_pro", "I think it is C") == "C"


def test_gsm_prediction_normalization():
    assert normalize_prediction("gsm_hard", "1,234") == "1234"
    assert normalize_prediction("gsm_hard", "#### -9867630.0") == "-9867630.0"
    assert normalize_prediction("gsm_hard", "The final answer is 42.") == "42"
    assert exact_match("gsm_hard", "5.97490e-6", "0.0000059749")


if __name__ == "__main__":
    test_mmlu_answer_normalization()
    test_mmlu_prediction_normalization()
    test_gsm_prediction_normalization()
