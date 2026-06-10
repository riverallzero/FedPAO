# FedPAO Orchestrator 구성 문서

검토일: 2026-06-10 KST

이 문서는 현재 FedPAO 코드에서 사용할 수 있는 orchestrator adapter와 외부 MAS repository 연결 방식을 정리한다. 현재 registry에 등록된 baseline은 MASRouter와 GPT-Swarm이다.

## 1. 지원 Orchestrator

현재 지원 목록:

```text
masrouter
gptswarm
```

두 orchestrator는 각각 두 가지 mode로 실행할 수 있다.

- `native`: `third_party/`에 clone된 원본 MAS code path를 호출한다.
- `simulated`: dependency-light prompt wrapping adapter로 동작한다. smoke/debug 용도이며 baseline 성능 평가용이 아니다.

등록 위치:

```text
src/orchestrators/registry.py
```

Registry mapping:

```text
mode=native:
  masrouter -> NativeMASRouterOrchestrator
  gptswarm  -> NativeGPTSwarmOrchestrator

mode=simulated:
  masrouter -> MASRouterOrchestrator
  gptswarm  -> GPTSwarmOrchestrator
```

## 2. 외부 Repo 위치

외부 MAS baseline repo는 `third_party/` 아래에 둔다.

```text
third_party/masrouter
third_party/GPTSwarm
```

Clone 명령:

```bash
mkdir -p third_party

git clone --depth 1 \
  https://github.com/yanweiyue/masrouter.git \
  third_party/masrouter

git clone --depth 1 \
  https://github.com/metauto-ai/GPTSwarm.git \
  third_party/GPTSwarm
```

`third_party/`는 git에 포함하지 않는다.

상태 확인:

```bash
.venv/bin/python scripts/check_orchestrators.py
.venv/bin/python scripts/check_orchestrators.py --json
```

예상 출력:

```text
masrouter  OK
  repo: third_party/masrouter
gptswarm   OK
  repo: third_party/GPTSwarm
```

## 3. 내부 Adapter 구조

관련 파일:

```text
src/orchestrators/base.py
src/orchestrators/local_baselines.py
src/orchestrators/native_baselines.py
src/orchestrators/registry.py
src/orchestrators/__init__.py
```

핵심 객체:

```text
OrchestratorSpec      baseline repo metadata
OrchestratedPrompt    adapter가 생성한 prompt와 metadata
BaseOrchestrator      공통 interface
LocalBaselineOrchestrator
  MASRouterOrchestrator
  GPTSwarmOrchestrator
NativeMASRouterOrchestrator
NativeGPTSwarmOrchestrator
```

`create_orchestrator(name, mode=...)`가 mode에 맞는 registry에서 adapter class를 선택한다. `scripts/run.py`, `scripts/run_test.py`, `scripts/run_client_inference.py` 모두 이 entrypoint를 사용한다.

## 4. MASRouter Native 연결

원본:

```text
https://github.com/yanweiyue/masrouter
```

로컬 위치:

```text
third_party/masrouter
```

현재 adapter가 확인하는 파일:

```text
third_party/masrouter/Experiments/run_mmlu.py
third_party/masrouter/Experiments/run_gsm8k.py
```

Native entrypoint:

```text
MAR.MasRouter.mas_router.MasRouter.forward
```

연결 방식:

- MASRouter의 circular import를 피하기 위해 module load 순서를 bootstrap한다.
- MASRouter 내부 LLM registry를 FedPAO local frozen LLM callable로 patch한다.
- `MasRouter(max_agent=3, device=device).to(device).eval()`로 router를 생성한다.
- `round0_checkpoint_path`가 있으면 checkpoint의 `router_state_dict`를 로드한다.
- MMLU-Pro는 MMLU final node role을, GSM-Hard는 GSM8K final node role을 사용한다.
- inference metadata에는 `cost`, `agents_num`, `response` 등이 포함된다.

Round 0 native training에서는 `masrouter_round0.pt` checkpoint가 생성된다. 이후 round와 `scripts/run_test.py`는 직전 round manifest의 `checkpoint_path`를 다시 adapter에 넘겨 로드한다.

## 5. GPT-Swarm Native 연결

원본:

```text
https://github.com/metauto-ai/GPTSwarm
```

로컬 위치:

```text
third_party/GPTSwarm
```

현재 adapter가 확인하는 파일:

```text
third_party/GPTSwarm/swarm
third_party/GPTSwarm/experiments/run_mmlu.py
third_party/GPTSwarm/pyproject.toml
```

Native entrypoint:

```text
swarm.graph.swarm.Swarm.run
```

연결 방식:

- GPT-Swarm `LLMRegistry`에 `fedpao_local` backend를 등록한다.
- `LLMRegistry.get()`을 FedPAO local LLM wrapper 반환 함수로 patch한다.
- `FedPAOLocalLLM.gen()`은 messages를 prompt text로 변환한 뒤 `llm_generate(prompt)`를 호출한다.
- 현재 graph는 `["IO", "COT", "IO"]` agent 구성을 사용한다.
- MMLU-Pro는 GPT-Swarm domain `mmlu`, GSM-Hard는 adapter가 runtime 등록하는 `gsm_hard` prompt set으로 연결한다.
- checkpoint가 있으면 `connection_dist_state_dict`를 로드하고 edge probability 기반 graph를 realize한다.

Round 0 native training에서는 `gptswarm_edges_round0.pt` checkpoint가 생성된다.

## 6. Config와 실행 옵션

기본 설정 파일:

```text
configs/fedpao_six_clients.yaml
```

현재 client 구성:

```text
c1: masrouter + mmlu_pro
c2: masrouter + gsm_hard
c3: gptswarm  + mmlu_pro
c4: gptswarm  + gsm_hard
```

현재 local model 설정:

```yaml
model:
  backend: hf_local
  model_id: /data/digirl/checkpoints/Qwen3-4B-Instruct/
  device: cuda
  dtype: auto
```

Orchestrator 선택:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_client_inference.py --mas masrouter
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_client_inference.py --mas gptswarm
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_client_inference.py --mas masrouter gptswarm
```

Mode 선택:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_client_inference.py \
  --orchestrator-mode simulated \
  --max-samples-per-client 1
```

`scripts/run.py`와 `scripts/run_test.py`도 동일하게 `--mas`와 `--orchestrator-mode`를 받는다.

## 7. Standalone Inference JSONL

`scripts/run_client_inference.py` 기본 출력:

```text
outputs/client_inference_smoke.jsonl
```

각 row의 공통 필드:

```json
{
  "client_id": "c1",
  "mas": "masrouter",
  "dataset": "mmlu_pro",
  "dataset_name": "TIGER-Lab/MMLU-Pro",
  "partition_mode": "client_disjoint",
  "partition_key": "TIGER-Lab/MMLU-Pro",
  "partition_rank": 0,
  "partition_peers": 2,
  "split": "validation",
  "example_index": 0,
  "source_example_index": 123,
  "model_path": "/data/digirl/checkpoints/Qwen3-4B-Instruct/",
  "orchestrator_mode": "native",
  "orchestrator_metadata": {},
  "response": "...",
  "prediction": "A",
  "ground_truth": "A",
  "correct": true,
  "latency_sec": 1.2345
}
```

Prediction normalization:

- `mmlu_pro`: 응답에서 `A`-`J` 단일 letter를 추출한다.
- `gsm_hard`: 응답에서 마지막 numeric token을 추출한다.

MASRouter metadata 예시:

```json
{
  "orchestrator_metadata": {
    "adapter_mode": "native",
    "orchestrator": "masrouter",
    "orchestrator_display_name": "MASRouter",
    "orchestrator_repo_path": "third_party/masrouter",
    "orchestrator_repo_available": true,
    "orchestrator_missing_paths": [],
    "native_entrypoint": "MAR.MasRouter.mas_router.MasRouter.forward",
    "cost": 0.0,
    "agents_num": 2.0,
    "response": "..."
  }
}
```

GPT-Swarm metadata 예시:

```json
{
  "orchestrator_metadata": {
    "adapter_mode": "native",
    "orchestrator": "gptswarm",
    "orchestrator_display_name": "GPT-Swarm",
    "orchestrator_repo_path": "third_party/GPTSwarm",
    "orchestrator_repo_available": true,
    "orchestrator_missing_paths": [],
    "native_entrypoint": "swarm.graph.swarm.Swarm.run",
    "swarm_domain": "gsm_hard",
    "swarm_agents": ["IO", "COT", "IO"],
    "response": "..."
  }
}
```

Simulated mode metadata는 `adapter_mode: simulated_local`을 사용하며 native entrypoint는 포함하지 않는다.

## 8. Round Runner와 Test Runner에서의 사용

`scripts/run.py` round flow:

```text
round 0 native/artifact checkpoint 생성
  -> round N inference에서 직전 checkpoint 로드
  -> trajectory와 analysis 생성
  -> client soft prompt 생성 및 CMA-ES refinement
  -> prompt를 주입한 client 재학습
  -> FedAvg global prompt 생성
```

`scripts/run_test.py`는 저장된 round manifest를 읽고, 각 client checkpoint를 같은 adapter로 로드해 held-out test set에서 평가한다. `--prompt-mode manifest`가 기본값이며, 해당 round manifest에 `global_prompt`가 있으면 embedding soft prompt를 주입한다. prompt 없이 checkpoint만 비교하려면 `--prompt-mode none`을 사용한다.

## 9. 새 Orchestrator 추가 절차

1. `BaseOrchestrator`를 상속한 simulated adapter를 추가한다.
2. 필요하면 native adapter를 `src/orchestrators/native_baselines.py`에 추가한다.
3. `OrchestratorSpec`에 repo dir, URL, expected paths를 정의한다.
4. `run_inference()`에서 `OrchestratedPrompt`와 metadata를 반환한다.
5. `ORCHESTRATORS` 또는 `NATIVE_ORCHESTRATORS` registry에 등록한다.
6. `SUPPORTED_MAS`와 config의 `clients[].mas`에 새 이름을 반영한다.

## 10. Caveats

- Native mode는 외부 repo import와 dependency 상태에 민감하다.
- `run_client_inference.py`는 adapter 내부 예외를 JSONL error row로 변환하지 않고 전체 실행을 중단한다.
- 세부 agent/tool call trajectory는 아직 baseline별 metadata 중심이며 완전히 통일된 schema는 아니다.
- `round0_checkpoint_path`라는 인자명은 round 1 이후에도 쓰이지만 실제로는 직전 round client checkpoint path를 의미한다.
