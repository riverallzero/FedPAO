# FedPAO 환경 구성 문서

검토일: 2026-06-10 KST

이 문서는 현재 코드 기준으로 FedPAO를 단일 머신에서 실행하기 위한 환경, 데이터, 모델, 스크립트 구성을 정리한다. 현재 기본 실험은 MASRouter와 GPT-Swarm 4개 client를 사용한다.

## 1. 현재 구현 상태

현재 구현된 항목:

- 4-client 기본 config: `configs/fedpao_six_clients.yaml`
- MASRouter, GPT-Swarm simulated/native orchestrator adapter
- round 0 native/artifact checkpoint 생성
- round 1 이후 validation inference, trajectory 생성, summary 분석
- summary report에서 soft prompt latent 생성
- CMA-ES 기반 client prompt refinement
- FedAvg 기반 server global prompt aggregation
- causal LM backend에 embedding-level soft prompt prefix 주입
- 완료된 run의 held-out test evaluation: `scripts/run_test.py`
- dataset partition mode: `shared`, `client_disjoint`

현재 기본 config는 같은 dataset을 쓰는 client끼리 서로 다른 shuffled subset을 쓰도록 `data.partition_mode: client_disjoint`를 사용한다.

## 2. 프로젝트 구조

```text
configs/             FedPAO experiment configs
docs/                환경, orchestrator, round-flow 문서
scripts/run.py       round 0 학습과 federated round 실행
scripts/run_test.py  저장된 round checkpoint held-out test 평가
scripts/run_client_inference.py
                     단일 MAS baseline inference smoke test
src/orchestrators/   MASRouter, GPT-Swarm adapters
third_party/         외부 MAS repositories, git ignored
outputs/runs/        run outputs, manifests, checkpoints, artifacts
```

`outputs/`, `third_party/`, local cache, virtualenv는 git에 포함하지 않는다.

## 3. Python 환경

권장 흐름:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip setuptools wheel
```

PyTorch는 CUDA 버전에 맞는 wheel을 먼저 설치한다. 예를 들어 CUDA 12.6 wheel을 쓰는 경우:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

그 뒤 FedPAO dependency를 설치한다.

```bash
pip install -r requirements.txt
```

환경 점검:

```bash
.venv/bin/python scripts/check_env.py
.venv/bin/python -m pip check
```

Codex sandbox 내부에서는 GPU device node가 보이지 않아 `torch.cuda.is_available()`가 `False`가 될 수 있다. 실제 GPU 검증과 긴 실험은 sandbox 밖 shell에서 실행하는 것이 안전하다.

## 4. 환경변수

`.env`가 있으면 `python-dotenv`로 로드된다. 예시:

```text
HF_HOME=./data/cache/huggingface
HF_DATASETS_CACHE=./data/cache/huggingface/datasets
TRANSFORMERS_CACHE=./data/cache/huggingface/transformers
TOKENIZERS_PARALLELISM=false
WANDB_PROJECT=fedpao
WANDB_MODE=offline
MLFLOW_TRACKING_URI=./mlruns
```

현재 기본 backend는 local Hugging Face checkpoint를 사용하는 `hf_local`이다. OpenAI API key는 기본 실행에 필요하지 않다.

## 5. 데이터

기본 config가 사용하는 dataset:

```text
TIGER-Lab/MMLU-Pro
reasoning-machines/gsm-hard
```

dataset 로드 점검:

```bash
.venv/bin/python scripts/check_datasets.py
```

기본 data 설정:

```yaml
data:
  partition_mode: client_disjoint
  train_fraction: 0.8
  max_samples_per_client: 100
  max_trajectory_samples_per_client: 3
```

`client_disjoint`에서는 같은 `dataset_name`을 공유하는 client들이 동일 shuffle을 기준으로 서로 겹치지 않는 구간을 받는다. 예를 들어 MMLU-Pro를 쓰는 MASRouter client와 GPT-Swarm client는 기본적으로 서로 다른 100개 예제를 사용한다. 이전처럼 같은 subset으로 비교하려면 config를 `shared`로 바꾸거나 실행 시 `--data-partition-mode shared`를 넘긴다.

## 6. 로컬 모델

기본 모델 경로는 config의 `model.model_id`에서 읽는다.

```yaml
model:
  backend: hf_local
  model_id: /data/digirl/checkpoints/Qwen3-4B-Instruct/
  device: cuda
  dtype: auto
```

실행 시 다른 checkpoint를 쓰려면 `--model-path`를 넘긴다.

```bash
.venv/bin/python scripts/run.py \
  --model-path /path/to/local/hf/checkpoint \
  --start-round 0 \
  --rounds 4
```

`scripts/run_client_inference.py`와 `scripts/run_test.py`도 같은 방식으로 `--model-path`를 받는다.

## 7. 기본 Config

현재 `configs/fedpao_six_clients.yaml`의 client 구성:

```text
c1: masrouter + MMLU-Pro
c2: masrouter + GSM-Hard
c3: gptswarm  + MMLU-Pro
c4: gptswarm  + GSM-Hard
```

주요 학습/최적화 설정:

```yaml
prompt:
  type: soft_latent
  latent_dim: 64
  init_std: 0.02

optimizer:
  name: cma_es
  sigma0: 0.5
  population_size: 5
  local_iterations: 10
  local_batch_size: 16

federated:
  aggregation: fedavg
  weight_by: num_examples

objective:
  cost_weight: 0.0
  latency_weight: 0.0
  latent_norm_weight: 0.05
```

Config 확인:

```bash
.venv/bin/python - <<'PY'
import yaml

with open("configs/fedpao_six_clients.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print("num clients:", len(cfg["clients"]))
print([client["id"] for client in cfg["clients"]])
print([client["mas"] for client in cfg["clients"]])
PY
```

## 8. 외부 MAS Repo

외부 repo는 `third_party/` 아래에 둔다.

```bash
mkdir -p third_party

git clone --depth 1 \
  https://github.com/yanweiyue/masrouter.git \
  third_party/masrouter

git clone --depth 1 \
  https://github.com/metauto-ai/GPTSwarm.git \
  third_party/GPTSwarm
```

상태 점검:

```bash
.venv/bin/python scripts/check_orchestrators.py
.venv/bin/python scripts/check_orchestrators.py --json
```

현재 registry에 등록된 orchestrator는 `masrouter`, `gptswarm`이다.

## 9. 실행 가능한 스크립트

Round flow 실행:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run.py \
  --start-round 0 \
  --rounds 4
```

Artifact-only smoke test:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run.py \
  --round0-trainer artifact \
  --max-samples-per-client 1 \
  --start-round 0 \
  --rounds 1
```

Standalone baseline inference:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_client_inference.py --mas masrouter
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_client_inference.py --mas gptswarm
```

Held-out test evaluation:

```bash
python3 scripts/run_test.py --run-root outputs/runs/<run-name> --rounds 0 1 2 3

python3 scripts/run_test.py \
  --run-root outputs/runs/<run-name> \
  --test-samples-per-client 20
```

## 10. 주요 출력

Round run은 기본적으로 `outputs/runs/<run-name>/` 아래에 저장된다.

```text
outputs/runs/<run-name>/
  config.yaml
  manifest.json
  round_000/
    manifest.json
  round_001/
    inference.jsonl
    analysis_manifest.json
    manifest.json
    server/
      global_prompt_manifest.json
      global_prompt_round002.pt
```

`scripts/run_test.py`는 기본적으로 `<run-root>/test/` 아래에 아래 파일을 만든다.

```text
test_inference.jsonl
test_manifest.json
test_sets/
  <client>_<mas>_<dataset>.jsonl
```

## 11. Caveats

- `artifact` trainer는 실제 MAS 학습이 아니라 checkpoint/artifact 계약 확인용이다.
- embedding-level soft prompt injection은 causal LM backend에서만 지원한다.
- native mode는 외부 repo dependency와 runtime behavior의 영향을 받는다.
- `run_client_inference.py`는 adapter 내부 예외를 error row로 변환하지 않고 실행을 중단한다.
