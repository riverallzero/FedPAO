import argparse

import yaml
from dotenv import load_dotenv


def unique_datasets(config: dict) -> dict[str, str]:
    datasets = {}
    for client in config["clients"]:
        datasets[client["dataset"]] = client["dataset_name"]
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fedpao_six_clients.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", default="data/processed/splits")
    args = parser.parse_args()

    load_dotenv()

    from src.data.loader import get_dataset
    from src.data.splits import build_split_spec, save_split_spec

    with open(args.config) as file:
        config = yaml.safe_load(file)

    seed = args.seed if args.seed is not None else int(config.get("seed", 42))

    for dataset, dataset_name in unique_datasets(config).items():
        dataset_dict = get_dataset(dataset_name)
        spec = build_split_spec(dataset, dataset_name, dataset_dict, config, seed)
        path = save_split_spec(spec, args.output_dir)
        counts = {
            role: len(split["indices"])
            for role, split in spec["splits"].items()
        }
        print(f"{dataset}: {counts} -> {path}")


if __name__ == "__main__":
    main()
