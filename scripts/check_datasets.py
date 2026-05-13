try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from datasets import load_dataset


DATASETS = [
    "TIGER-Lab/MMLU-Pro",
    "reasoning-machines/gsm-hard",
]


for name in DATASETS:
    print("=" * 80)
    print("Loading:", name)
    dataset = load_dataset(name)
    print(dataset)

    first_split = list(dataset.keys())[0]
    print("First split:", first_split)
    print("First example:")
    print(dataset[first_split][0])
