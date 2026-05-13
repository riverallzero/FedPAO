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
python scripts/prepare_data_splits.py --config configs/fedpao_six_clients.yaml --seed 42
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
  --max-new-tokens 8 \
  --split-role eval
```

Validate causal soft-prompt injection:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/check_soft_prompt.py \
  --config configs/fedpao_six_clients.yaml \
  --max-new-tokens 2
```

Run the mock orchestrator with the zero latent:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_mock_orchestrator.py \
  --config configs/fedpao_six_clients.yaml \
  --max-samples-per-client 1 \
  --max-new-tokens 2
```

Run a minimal single-client CMA-ES smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_local_cma_smoke.py \
  --config configs/fedpao_six_clients.yaml \
  --client-id c1 \
  --max-train-samples 1 \
  --population-size 2 \
  --local-iterations 1 \
  --max-new-tokens 2
```

Run a minimal 6-client FedPAO dry run:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_fedpao_dryrun.py \
  --config configs/fedpao_smoke_taskonly.yaml \
  --rounds 1 \
  --clients-per-round 6 \
  --max-train-samples-per-client 1 \
  --max-eval-samples-per-client 1 \
  --population-size 2 \
  --local-iterations 1 \
  --max-new-tokens 2 \
  --eval-every-round
```

List local causal model candidates:

```bash
python scripts/list_model_candidates.py --root /data/digirl/checkpoints
```

## Notes

- Keep external MAS repositories under `third_party/`; they are intentionally git-ignored.
- Start with mock orchestrators before installing GPT-Swarm, MASRouter, and Puppeteer dependencies into this environment.
- The default development model is `/data/digirl/checkpoints/Qwen3.5-2B-Base` loaded through `AutoModelForCausalLM`.
- The earlier LLaVA-OneVision checkpoint remains in `model.smoke_model_id` for image-text smoke checks.
