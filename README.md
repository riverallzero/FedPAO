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
python3 -m pip install --upgrade pip setuptools wheel
```

Install PyTorch first using the wheel appropriate for the machine.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
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
python3 scripts/check_env.py
python3 scripts/check_datasets.py
```

Validate the 6-client config:

```bash
python3 - <<'PY'
import yaml

with open("configs/fedpao_six_clients.yaml") as f:
    cfg = yaml.safe_load(f)

print("num clients:", len(cfg["clients"]))
print([client["id"] for client in cfg["clients"]])
PY
```

Validate the cloned orchestrator baselines:

```bash
python3 scripts/check_orchestrators.py
```

Run one local-model inference sample for each configured client:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_client_inference.py \
  --max-samples-per-client 1 \
  --max-new-tokens 64 \
  --puppeteer-reward-model-path /data/Llama-3.1-Nemotron-70B-Reward-HF/ \
  --puppeteer-global-config configs/orchestrators/puppeteer_global.yaml \
  --puppeteer-policy-config configs/orchestrators/puppeteer_policy.json
```

Run only one orchestrator baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_client_inference.py --mas puppeteer
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_client_inference.py --mas masrouter
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_client_inference.py --mas gptswarm
```

## Notes

- Keep external MAS repositories under `third_party/`; they are intentionally git-ignored.
- By default, `scripts/run_client_inference.py` uses native adapters that call the cloned Puppeteer, MASRouter, and GPT-Swarm code paths.
- Puppeteer native mode reads FedPAO-owned config templates from `configs/orchestrators/puppeteer_global.yaml` and `configs/orchestrators/puppeteer_policy.json`, then projects them into the cloned repo's expected `puppeteer/config/` paths only while the run is active.
- Puppeteer reward model path is resolved from `--puppeteer-reward-model-path`, `configs/fedpao_six_clients.yaml`, `PUPPETEER_REWARD_MODEL_PATH`, or the default `/data/Llama-3.1-Nemotron-70B-Reward-HF/`.
- The FedPAO Puppeteer policy config is set to inference mode and the native adapter disables policy checkpoint load/save calls during smoke inference.
- Use `--orchestrator-mode simulated` only for dependency-light prompt wrapping smoke tests.
- Native adapters require each external MAS dependency stack to be installed or otherwise importable in the active environment.
- Replace `model.model_id` in `configs/fedpao_six_clients.yaml` before running local LLM experiments.
