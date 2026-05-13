# FedPAO 실험 환경 구성 문서

검증일: 2026-05-13 KST

이 문서는 현재 git 디렉토리(`/data/fl_agent/Fed-PAO`)에 구성된 FedPAO 실험 환경, 데이터, 로컬 모델, 실행 스크립트, 검증 결과를 정리한 것이다. 현재 단계의 목표는 full FedPAO 학습 코드를 완성하는 것이 아니라, 단일 머신에서 6개 client 실험을 반복 실행할 수 있는 기반을 만드는 것이다.

## 1. 현재 상태 요약

현재 완료된 항목:

- Python 가상환경 `.venv` 생성
- FedPAO 기본 dependency 설치
- PyTorch CUDA wheel 설치
- MMLU-Pro, GSM-Hard dataset cache 구성
- 6-client config 작성
- 로컬 모델 경로 설정
- 외부 MAS repo clone
- 6개 client별 local model inference smoke test 실행
- deterministic dataset split 파일 생성
- dataset loader / evaluator normalization 모듈 추가
- inference JSONL에 `example_id`, token count, mock call metric 기록
- Qwen3.5-2B-Base `AutoModelForCausalLM` 로딩 검증
- frozen causal LM soft prompt embedding injection smoke test 통과
- mock orchestrator / local client 평가 경로 구현
- 단일 client CMA-ES local optimization smoke test 통과
- 6-client 1-round FedAvg dry run 통과

아직 완료하지 않은 항목:

- Puppeteer / MASRouter / GPT-Swarm adapter 실연동
- full benchmark evaluation loop 구현
- full-size hyperparameter 실험

## 2. 프로젝트 루트와 디렉토리 구조

프로젝트 루트:

```bash
/data/fl_agent/Fed-PAO
```

주요 디렉토리:

```text
src/clients        client별 local optimization 코드 위치
src/server         FedAvg, aggregation, round control 코드 위치
src/orchestrators  Puppeteer, MASRouter, GPT-Swarm wrapper 위치
src/prompts        soft prompt / latent prompt 처리 위치
src/optimizers     CMA-ES optimizer 코드 위치
src/evaluation     accuracy, cost, latency, redundant call metric 위치
src/utils          공통 유틸 위치
configs            FedPAO, client, model 설정
data/cache         Hugging Face dataset/model cache
outputs            inference/evaluation 결과
logs               실행 로그
scripts            환경 점검 및 smoke test 스크립트
third_party        외부 MAS repo clone 위치
```

Git ignore 대상:

```text
.venv/
.env
data/cache/
data/raw/
data/processed/* except data/processed/splits/*.json
outputs/
logs/
wandb/
mlruns/
third_party/
```

## 3. Python 환경

가상환경:

```bash
/data/fl_agent/Fed-PAO/.venv
```

`python3 -m venv`는 시스템에 `python3.10-venv`가 없어 실패했고, 사용자 영역의 `virtualenv`로 `.venv`를 생성했다.

활성화:

```bash
cd /data/fl_agent/Fed-PAO
source .venv/bin/activate
```

검증된 주요 버전:

```text
python        3.10.12
torch         2.11.0+cu128
torch cuda    12.8
transformers  5.8.0
datasets      4.8.5
accelerate    1.13.0
evaluate      0.4.6
cma           4.4.4
flwr          1.29.0
openai        2.36.0
mlflow        3.12.0
wandb         0.26.1
```

Dependency consistency:

```bash
.venv/bin/python -m pip check
```

검증 결과:

```text
No broken requirements found.
```

## 4. GPU / CUDA 상태

현재 머신에는 RTX 6000 Ada Generation GPU 4개가 보인다.

PyTorch CUDA 검증 결과:

```text
torch.cuda.is_available(): True
torch.cuda.device_count(): 4
torch.cuda.get_device_name(0): NVIDIA RTX 6000 Ada Generation
```

`nvidia-smi` 기준 GPU:

```text
0, NVIDIA RTX 6000 Ada Generation, 49140 MiB
1, NVIDIA RTX 6000 Ada Generation, 49140 MiB
2, NVIDIA RTX 6000 Ada Generation, 49140 MiB
3, NVIDIA RTX 6000 Ada Generation, 49140 MiB
```

주의:

Codex sandbox 내부에서는 `/dev/nvidia*` device node가 보이지 않아 `torch.cuda.is_available()`가 `False`가 될 수 있다. 실제 GPU 검증은 sandbox 밖 실행 권한으로 확인했다.

## 5. 환경변수

예시 파일:

```bash
.env.example
```

로컬 파일:

```bash
.env
```

현재 기본 설정:

```text
HF_HOME=./data/cache/huggingface
HF_DATASETS_CACHE=./data/cache/huggingface/datasets
TRANSFORMERS_CACHE=./data/cache/huggingface/transformers
TOKENIZERS_PARALLELISM=false
WANDB_PROJECT=fedpao
WANDB_MODE=offline
MLFLOW_TRACKING_URI=./mlruns
```

API 기반 실험을 하려면 `.env`에 다음 값을 채운다.

```text
OPENAI_API_KEY=
OPENAI_MODEL=
```

현재 본 scaffold는 `hf_local` backend를 우선 사용하므로 OpenAI API key는 필수값이 아니다.

## 6. Dataset 상태

사용 dataset:

```text
TIGER-Lab/MMLU-Pro
reasoning-machines/gsm-hard
```

캐시 위치:

```text
data/cache/huggingface/datasets/TIGER-Lab___mmlu-pro
data/cache/huggingface/datasets/reasoning-machines___gsm-hard
```

캐시 파일:

```text
mmlu-pro-test.arrow
mmlu-pro-validation.arrow
gsm-hard-train.arrow
dataset_info.json
```

Offline mode 검증 결과:

```text
TIGER-Lab/MMLU-Pro {'test': 12032, 'validation': 70}
reasoning-machines/gsm-hard {'train': 1319}
```

즉, 현재 두 dataset은 네트워크 없이도 로드 가능하다.

Dataset smoke test:

```bash
source .venv/bin/activate
python scripts/check_datasets.py
```

## 7. 로컬 모델 설정

로컬 checkpoint root:

```bash
/data/digirl/checkpoints
```

현재 config에 설정된 모델:

```bash
/data/digirl/checkpoints/Qwen3.5-2B-Base
```

이 모델을 선택한 이유:

- 현재 설치된 `transformers`에서 `Qwen3_5ForCausalLM`로 로딩 가능하다.
- MMLU-Pro / GSM-Hard는 텍스트 benchmark라 causal LM backend가 soft prompt injection 구현에 더 적합하다.
- `/data/digirl/checkpoints` 안에서 soft prompt 개발에 쓸 수 있는 2B급 후보로 확인됐다.

이전 smoke test용 모델:

```bash
/data/digirl/checkpoints/llava-onevision-qwen2-0.5b-ov-hf
```

이 모델은 `model.smoke_model_id`에 남겨 두었다.

간단 로딩 검증:

```bash
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "/data/digirl/checkpoints/Qwen3.5-2B-Base"
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    path,
    local_files_only=True,
    dtype=torch.float16,
    device_map="auto",
).eval()

print(type(model).__name__)
PY
```

다른 checkpoint 후보는 많지만, 아직 FedPAO config에는 연결하지 않았다. 예:

```text
Qwen3.5-2B
Qwen3.5-2B-Base
Qwen2-VL-2B-Instruct
Qwen2-VL-7B-Instruct
llava-1.5-7b
llava-1.6-7b
InternVL3_5-1B
InternVL3_5-8B
Ovis2-1B
Phi-4-multimodal-instruct
```

## 8. 6-Client FedPAO Config

설정 파일:

```bash
configs/fedpao_six_clients.yaml
```

핵심 설정:

```yaml
model:
  backend: hf_causal
  model_id: /data/digirl/checkpoints/Qwen3.5-2B-Base
  smoke_model_id: /data/digirl/checkpoints/llava-onevision-qwen2-0.5b-ov-hf
  device: cuda
  dtype: auto

prompt:
  type: soft_latent
  latent_dim: 32
  prompt_length: 8
  projection_seed: 42
  init_std: 0.02

optimizer:
  name: cma_es
  sigma0: 0.5
  population_size: 8
  local_iterations: 5
  local_batch_size: 16

federated:
  rounds: 10
  clients_per_round: 6
  aggregation: fedavg
  weight_by: num_examples
```

Client 구성:

```text
c1: puppeteer + MMLU-Pro
c2: puppeteer + GSM-Hard
c3: masrouter  + MMLU-Pro
c4: masrouter  + GSM-Hard
c5: gptswarm   + MMLU-Pro
c6: gptswarm   + GSM-Hard
```

Config 검증:

```bash
python - <<'PY'
import yaml

with open("configs/fedpao_six_clients.yaml") as f:
    cfg = yaml.safe_load(f)

print("num clients:", len(cfg["clients"]))
print([client["id"] for client in cfg["clients"]])
PY
```

예상 출력:

```text
num clients: 6
['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
```

## 9. 외부 MAS Repo

외부 repo는 clone만 했고, 현재 `.venv`에 설치하지 않았다.

```text
third_party/GPTSwarm
third_party/masrouter
third_party/ChatDev-puppeteer
```

설치하지 않은 이유:

- 각 MAS repo가 요구하는 `torch`, `transformers`, `openai`, `numpy` 버전이 현재 FedPAO 환경과 충돌할 수 있다.
- 특히 ChatDev Puppeteer requirements는 `torch==2.4.0`, `transformers==4.43.4` 등 고정 버전을 요구한다.
- 초기 단계에서는 FedPAO wrapper interface를 먼저 만들고, 각 MAS는 adapter로 하나씩 연결하는 편이 안전하다.

권장 연결 순서:

```text
1. FedPAO mock/raw orchestrator
2. GPTSwarm wrapper
3. MASRouter wrapper
4. Puppeteer wrapper
```

## 10. 현재 실행 가능한 스크립트

환경 점검:

```bash
python scripts/check_env.py
```

Dataset 점검:

```bash
python scripts/check_datasets.py
```

Dataset split 생성:

```bash
python scripts/prepare_data_splits.py --config configs/fedpao_six_clients.yaml --seed 42
```

로컬 모델 후보 점검:

```bash
python scripts/list_model_candidates.py --root /data/digirl/checkpoints
```

Soft prompt injection 점검:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/check_soft_prompt.py \
  --config configs/fedpao_six_clients.yaml \
  --max-new-tokens 2
```

Mock orchestrator 점검:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_mock_orchestrator.py \
  --config configs/fedpao_six_clients.yaml \
  --max-samples-per-client 1 \
  --max-new-tokens 2
```

CMA-ES local optimization smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_local_cma_smoke.py \
  --config configs/fedpao_six_clients.yaml \
  --client-id c1 \
  --max-train-samples 1 \
  --population-size 2 \
  --local-iterations 1 \
  --max-new-tokens 2
```

6-client FedPAO dry run:

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

6-client local inference smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_client_inference.py \
  --max-samples-per-client 1 \
  --max-new-tokens 8 \
  --split-role eval
```

기본 출력:

```bash
outputs/client_inference_smoke.jsonl
```

## 11. Local Inference Smoke Test 결과

실행한 명령:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python scripts/run_client_inference.py \
  --max-samples-per-client 1 \
  --max-new-tokens 8 \
  --split-role eval \
  --output outputs/client_inference_phase1.jsonl
```

결과 요약:

```text
c1 puppeteer mmlu_pro: pred='B' gt='A' correct=False
c2 puppeteer gsm_hard: free-form wrong response, gt='9.0' correct=False
c3 masrouter mmlu_pro: pred='B' gt='A' correct=False
c4 masrouter gsm_hard: free-form wrong response, gt='9.0' correct=False
c5 gptswarm mmlu_pro: pred='B' gt='A' correct=False
c6 gptswarm gsm_hard: free-form wrong response, gt='9.0' correct=False
accuracy: 0/6
```

해석:

- 이 결과는 모델 성능 평가가 아니라 local model inference pipeline smoke test다.
- 현재는 실제 Puppeteer / MASRouter / GPT-Swarm reasoning loop가 붙지 않았다.
- 각 client는 동일 로컬 모델을 raw text prompt로 호출한다.
- JSONL에는 `example_id`, `prompt_tokens`, `generated_tokens`, `total_tokens`, `agent_calls`, `tool_calls`가 포함된다.

생성된 split 파일:

```text
data/processed/splits/mmlu_pro_seed42.json
data/processed/splits/gsm_hard_seed42.json
```

Split count:

```text
mmlu_pro: train=49, eval=21, test=12032
gsm_hard: train=923, eval=198, test=198
```

Soft prompt smoke test 결과:

```text
model_class: Qwen3_5ForCausalLM
hidden_size: 2048
soft_prompt_shape: (8, 2048)
parameters_frozen: True
base_generation: 'A'
soft_generation: 'A'
```

CMA/FedAvg smoke 결과:

```text
single-client CMA:
  client_id: c1
  num_evaluations: 3
  best_z_norm: 0.0000

6-client FedPAO dry run:
  rounds: 1
  clients: 6
  population_size: 2
  local_iterations: 1
  num_evaluations_per_client: 3
  global_z_norm: 0.0000
  client_upload_bytes: 768
  server_download_bytes: 768
  total_round_bytes: 1536
```

확장 smoke 결과:

```text
command: configs/fedpao_smoke_taskonly.yaml, rounds=2, clients=6, train samples=4, eval samples=4, population=4, local_iterations=2
client_metrics rows: 12
round metrics rows: 2
eval_metrics rows: 12
num_evaluations_per_client: 9
round link check: round_001/global_latent_input.npy == round_000/global_latent_output.npy
finite check: pass
```

주의:

```text
task-only exact-match objective에서는 모든 후보가 동률이면 initial latent가 유지된다.
따라서 이 smoke에서 global_z_norm=0.0인 것은 성능 개선 실패가 아니라 baseline retention 동작 확인으로 해석한다.
```

Baseline smoke:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/run_latent_baselines.py \
  --config configs/fedpao_smoke_taskonly.yaml \
  --clients-per-round 2 \
  --max-train-samples-per-client 1 \
  --max-eval-samples-per-client 1 \
  --population-size 2 \
  --local-iterations 1 \
  --max-new-tokens 2 \
  --include-local-cma \
  --output-dir outputs/latent_baselines_smoke
```

결과:

```text
baseline_metrics rows: 8
baselines: no_soft_prompt, zero_latent, random_latent, local_only_cma
local_only_cma num_evaluations: 3
```

## 12. 학습을 해야 하는가?

지금 단계에서 LLM 자체 fine-tuning은 하지 않는 것이 맞다.

프로포절 기준 FedPAO의 핵심 설정은 다음이다.

```text
LLM parameters: frozen
gradient access: not used
optimization: CMA-ES
optimized variable: low-dimensional prompt latent z
aggregation: FedAvg-style aggregation over client prompt latents
```

따라서 여기서 말하는 “학습”은 일반적인 model fine-tuning이 아니라 다음 단계다.

```text
1. 각 client가 frozen local LLM을 호출한다.
2. CMA-ES가 client별 prompt latent 후보를 생성한다.
3. 후보 prompt latent의 score를 benchmark metric으로 평가한다.
4. local best latent를 server로 보낸다.
5. server가 FedAvg로 global latent를 aggregate한다.
6. 다음 federated round에 global latent를 broadcast한다.
```

즉, 현재 환경에서는 먼저 inference가 동작하는지 확인했고, 다음 구현 대상은 CMA-ES local optimizer와 FedAvg round loop다.

## 13. 다음 구현 순서

권장 순서:

```text
1. Broaden dry run sample size and iterations
2. Add focused tests for FedAvg, split loading, and CMA wrapper
3. Add MLflow/W&B offline logging hooks
4. Add dense scoring for MMLU option log-likelihood
5. GPTSwarm wrapper
6. MASRouter wrapper
7. Puppeteer wrapper
8. cost / latency / redundant call metric refinement
9. full FedPAO experiment
```

## 14. Known Caveats

- `outputs/`, `data/cache/`, `third_party/`는 git ignore 대상이다.
- 현재 inference 결과 파일은 재현용 artifact이며 git에는 포함하지 않는다.
- Codex sandbox 내부에서는 CUDA device node가 보이지 않을 수 있으므로 GPU 검증은 sandbox 밖 실행으로 해야 한다.
- 외부 MAS repo는 clone되어 있지만 설치하지 않았다.
- 현재 local inference는 adapter 없는 raw/mock prompt 방식이다.
- embedding-level soft prompt injection은 mock orchestrator와 CMA-ES/FedAvg dry run에 연결됐지만, 아직 실제 MAS adapter에는 연결되지 않았다.
- 현재 CMA/FedAvg 검증은 smoke test라 성능 의미는 없다. task-only objective에서 모든 후보가 동률이면 initial global latent가 baseline으로 유지되어 `z_norm=0`이 정상적으로 나올 수 있다.
- API backend는 설정만 존재하고 smoke test는 수행하지 않았다.
