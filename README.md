# Fed-PAO

Single-machine scaffold for the initial FedPAO experiments:

- black-box/frozen LLM backend
- CMA-ES local prompt latent optimization
- FedAvg-style federated aggregation
- 6 simulated clients: Puppeteer, MASRouter, GPT-Swarm crossed with MMLU-Pro and GSM-Hard

See [docs/environment_setup.md](docs/environment_setup.md) for the verified local environment, dataset cache, model path, and smoke test status.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch first using the wheel appropriate for the machine. For a CPU-only smoke test:

```bash
pip install torch torchvision torchaudio
```

Then install the FedPAO dependencies:

```bash
pip install -r requirements.txt
```

Create local environment variables:

```bash
cp .env.example .env
set -a
source .env
set +a
```

## Smoke Tests

```bash
python scripts/check_env.py
python scripts/check_datasets.py
```

Validate the 6-client config:

```bash
python - <<'PY'
import yaml

with open("configs/fedpao_six_clients.yaml") as f:
    cfg = yaml.safe_load(f)

print("num clients:", len(cfg["clients"]))
print([client["id"] for client in cfg["clients"]])
PY
```

Run one local-model inference sample for each configured client:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_client_inference.py \
  --max-samples-per-client 1 \
  --max-new-tokens 64
```

## Notes

- Keep external MAS repositories under `third_party/`; they are intentionally git-ignored.
- Start with mock orchestrators before installing GPT-Swarm, MASRouter, and Puppeteer dependencies into this environment.
- Replace `model.model_id` in `configs/fedpao_six_clients.yaml` before running local LLM experiments.
