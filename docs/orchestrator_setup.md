# FedPAO Orchestrator 구성 문서

작성일: 2026-05-15 KST

이 문서는 FedPAO에서 6개 simulated client가 사용하는 orchestrator baseline 연결 상태를 정리한다. 현재 구현은 `third_party/`에 clone한 Puppeteer, MASRouter, GPT-Swarm의 native code path를 직접 호출하고, baseline 내부 LLM 호출을 FedPAO local frozen LLM callable로 우회시켜 smoke inference를 수행하는 단계다.

현재 native mode는 full FedPAO 성능 평가가 아니라 "외부 MAS runner + 로컬 LLM backend + client별 dataset 입력"이 한 프로세스에서 연결되는지 확인하는 통합 smoke test에 가깝다.

## 1. 현재 상태 요약

완료된 항목:

- Puppeteer, MASRouter, GPT-Swarm repo를 `third_party/` 아래에 두는 구조 정의
- `src/orchestrators` 공통 adapter interface 작성
- 3개 baseline에 대한 simulated adapter와 native adapter 작성
- `scripts/run_client_inference.py`가 client별 `mas` 값에 따라 adapter를 생성하도록 연결
- 기본 실행 mode를 `native`로 설정
- `--orchestrator-mode simulated`로 dependency-light prompt wrapping만 확인 가능
- `--mas puppeteer|masrouter|gptswarm` 옵션으로 baseline별 부분 실행 가능
- Puppeteer reward model, global config, policy config를 FedPAO `configs/` 아래에서 관리
- Puppeteer native 실행 중 third_party config path에 FedPAO config를 임시 투영하고 정상 예외 흐름에서 원복
- Puppeteer import-time OpenAI client 생성을 dummy class로 우회
- Puppeteer `query_manager.query`를 FedPAO local LLM callable로 패치
- Puppeteer policy checkpoint load/save 호출을 smoke inference 중 비활성화
- MASRouter circular import 우회, local LLM registry patch, `torch.no_grad()` inference 적용
- GPT-Swarm `LLMRegistry`에 `fedpao_local` backend 등록 및 `LLMRegistry.get` patch 적용
- GPT-Swarm GSM-Hard용 `gsm_hard` prompt set runtime 등록
- baseline repo 존재 여부와 필수 파일을 확인하는 `scripts/check_orchestrators.py` 작성
- smoke inference 결과를 JSONL로 저장하고 `prediction`, `ground_truth`, `correct`, `latency_sec`, `orchestrator_metadata` 기록

아직 보강할 항목:

- baseline별 trajectory, agent call, tool call metric 수집
- FedPAO objective의 redundant call, invalid action, communication cost와 orchestrator trace 연결
- baseline 실행 실패를 client/example 단위 error metadata로 남기고 전체 run을 계속하는 fault-tolerant 실행
- baseline별 output을 통일된 FedPAO trajectory schema로 변환
- baseline별 dependency conflict가 커질 경우 subprocess 또는 별도 environment runner로 분리

## 2. 외부 Repo 위치

외부 MAS baseline repo는 모두 `third_party/` 아래에 둔다.

```text
third_party/ChatDev-puppeteer
third_party/masrouter
third_party/GPTSwarm
```

Clone 명령:

```bash
mkdir -p third_party

git clone --branch puppeteer --depth 1 \
  https://github.com/OpenBMB/ChatDev.git \
  third_party/ChatDev-puppeteer

git clone --depth 1 \
  https://github.com/yanweiyue/masrouter.git \
  third_party/masrouter

git clone --depth 1 \
  https://github.com/metauto-ai/GPTSwarm.git \
  third_party/GPTSwarm
```

`third_party/`는 FedPAO git에 포함하지 않는다. 새 환경에서는 위 clone 명령으로 다시 구성한다.

## 3. FedPAO 내부 Adapter 구조

현재 orchestrator 관련 코드는 다음 위치에 있다.

```text
src/orchestrators/base.py
src/orchestrators/local_baselines.py
src/orchestrators/native_baselines.py
src/orchestrators/registry.py
src/orchestrators/__init__.py
```

핵심 구조:

```text
OrchestratorSpec      baseline repo metadata
OrchestratedPrompt    adapter가 생성한 prompt와 metadata
BaseOrchestrator      공통 interface
LocalBaselineOrchestrator
  PuppeteerOrchestrator
  MASRouterOrchestrator
  GPTSwarmOrchestrator
NativePuppeteerOrchestrator
NativeMASRouterOrchestrator
NativeGPTSwarmOrchestrator
```

`scripts/run_client_inference.py`의 기본 흐름:

```text
load config
load local HF model backend
for each selected client:
  create_orchestrator(client["mas"], mode=args.orchestrator_mode)
  load client dataset
  build task prompt and ground truth
  pass llm_generate(prompt) callable into native adapter
  adapter calls baseline native runner
  write one JSONL row per example
```

기본 adapter mode는 `native`이다.

```bash
python3 scripts/run_client_inference.py
```

Dependency-light prompt wrapping만 확인하고 싶을 때는 simulated mode를 명시한다.

```bash
python3 scripts/run_client_inference.py --orchestrator-mode simulated
```

`simulated` mode는 실제 multi-agent runner를 실행하지 않으므로 baseline 동작 확인이나 성능 평가에 사용하면 안 된다.

## 4. Baseline별 Native 연결

### Puppeteer

원본:

```text
https://github.com/OpenBMB/ChatDev/tree/puppeteer
```

로컬 위치:

```text
third_party/ChatDev-puppeteer
```

현재 adapter가 확인하는 파일:

```text
third_party/ChatDev-puppeteer/puppeteer/main.py
third_party/ChatDev-puppeteer/requirements.txt
```

Native entrypoint:

```text
puppeteer.tasks.runner.BenchmarkRunner.run_reasoning
```

현재 연결 방식:

- `third_party/ChatDev-puppeteer/puppeteer`를 import path와 cwd로 사용한다.
- Puppeteer import 중 실제 OpenAI client가 초기화되지 않도록 `openai.OpenAI`를 임시 dummy class로 패치한다.
- Puppeteer 내부 `model.query_manager.query`를 `llm_generate(prompt)` 호출로 패치한다.
- `MMLU-Pro`와 `GSM-Hard` task dict를 만들어 `BenchmarkRunner.run_reasoning`에 전달한다.
- Puppeteer가 기대하는 `puppeteer/config/global.yaml`, `puppeteer/config/policy.json`에는 FedPAO-owned config를 실행 중에만 임시 투영한다.
- 실행 후 정상 예외 흐름에서는 third_party 원본 config 파일 내용을 복구한다.
- `ContinuousREINFORCE.load_model`, `ContinuousREINFORCE.save_model`을 no-op 처리한다.
- Puppeteer 전체 checkpoint/log/artifact 저장을 전역 차단한 것은 아니다. 현재 policy config에는 `paths.checkpoint_path: checkpoint/fedpao_inference`가 남아 있다.

FedPAO에서 관리하는 Puppeteer 설정:

```text
configs/orchestrators/puppeteer_global.yaml
configs/orchestrators/puppeteer_policy.json
```

`configs/orchestrators/puppeteer_global.yaml`의 주요 값:

```yaml
model_weight_path: /data/Llama-3.1-Nemotron-70B-Reward-HF/
external_tools_enabled: false
max_retry_times: 10
max_json_reformat_turns: 10
graph:
  max_parallel_paths: 2
  max_step_num: 4
```

`configs/orchestrators/puppeteer_policy.json`의 inference 관련 값:

```json
{
  "paths": {
    "checkpoint_path": "checkpoint/fedpao_inference"
  },
  "training": {
    "loading": false,
    "training": false
  },
  "agent": {
    "max_num_agents": 3,
    "next_num_agents": 3
  }
}
```

Reward model 경로 우선순위:

```text
1. --puppeteer-reward-model-path
2. configs/fedpao_six_clients.yaml orchestrators.puppeteer.reward_model_path
3. PUPPETEER_REWARD_MODEL_PATH
4. /data/Llama-3.1-Nemotron-70B-Reward-HF/
```

주의:

- Puppeteer 원본은 별도 dependency와 환경을 가정하지만, 현재 FedPAO에서는 필요한 dependency를 `requirements.txt`에 누적해 같은 `.venv`에서 native smoke를 시도한다.
- Puppeteer config의 원본은 FedPAO `configs/orchestrators/`이다. third_party config는 runtime compatibility를 위한 투영 대상이다.
- process가 강제 종료되면 context manager의 복구가 실행되지 않을 수 있으므로, 비정상 종료 후에는 third_party config 변경 여부를 확인한다.

### MASRouter

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

현재 연결 방식:

- MASRouter의 `MAR.Agent.agent_registry`와 `MAR.Graph.graph` 사이 circular import를 피하기 위해 `bootstrap_masrouter_imports()`에서 module load 순서를 고정한다.
- `LLMRegistry.get`을 `FedPAOLocalLLM` 반환 함수로 패치한다.
- `FedPAOLocalLLM.model_name`을 제공해 MASRouter 내부 logging과 final node 참조가 실패하지 않게 한다.
- `MasRouter(max_agent=3, device=device).to(device).eval()`로 router를 생성한다.
- `torch.no_grad()` 안에서 `router.forward(...)`를 호출한다. `torch.inference_mode()`는 MASRouter 내부 autograd graph 구성과 충돌할 수 있어 사용하지 않는다.
- MMLU-Pro는 `task_label=1`, `MAR/Roles/FinalNode/mmlu.json`을 사용한다.
- GSM-Hard는 `task_label=0`, `MAR/Roles/FinalNode/gsm8k.json`을 사용한다.
- device는 `torch.cuda.is_available()` 기준으로 `cuda` 또는 `cpu`를 자동 선택한다.

주의:

- MASRouter 원본은 자체 dataset file layout과 API backend 설정을 가정한다.
- 현재 adapter는 Hugging Face dataset example을 query string으로 변환해 `forward`에 직접 전달한다.
- 현재 output metadata는 `cost`, `agents_num`, `response` 중심이며, 세부 routing trajectory는 아직 통일 schema로 저장하지 않는다.

### GPT-Swarm

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

현재 연결 방식:

- GPT-Swarm `LLMRegistry`에 `fedpao_local` LLM class를 등록한다.
- GPT-Swarm 기본 `LLMRegistry.get()`은 `mock`이 아닌 model name을 모두 `GPTChat`으로 보내므로, native adapter에서 `LLMRegistry.get` 자체를 `FedPAOLocalLLM` 반환 함수로 패치한다.
- `FedPAOLocalLLM.gen()`은 messages를 prompt text로 변환한 뒤 FedPAO `llm_generate(prompt)`를 호출한다.
- 현재 graph는 smoke용으로 `["IO", "COT", "IO"]`를 사용한다.
- MMLU-Pro는 GPT-Swarm 내부 domain `mmlu`로 연결한다.
- GSM-Hard는 adapter가 runtime 등록하는 `gsm_hard` prompt set으로 연결한다.
- `edge_optimize=False`로 실행해 inference smoke 중 graph edge 최적화 학습을 수행하지 않는다.

주의:

- 원본 README는 Poetry 기반 설치 흐름을 사용한다.
- 현재 graph template은 smoke 기준이며, full evaluation 전에는 task별 graph config를 별도 설정으로 분리하는 것이 좋다.

## 5. 6-Client Config와 실행 옵션

설정 파일:

```text
configs/fedpao_six_clients.yaml
```

현재 local model 설정:

```yaml
model:
  backend: hf_local
  model_id: /data/digirl/checkpoints/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/
  device: cuda
  dtype: auto
```

Client 구성:

```text
c1: puppeteer + mmlu_pro
c2: puppeteer + gsm_hard
c3: masrouter  + mmlu_pro
c4: masrouter  + gsm_hard
c5: gptswarm   + mmlu_pro
c6: gptswarm   + gsm_hard
```

Orchestrator config:

```yaml
orchestrators:
  puppeteer:
    reward_model_path: /data/Llama-3.1-Nemotron-70B-Reward-HF/
    global_config: configs/orchestrators/puppeteer_global.yaml
    policy_config: configs/orchestrators/puppeteer_policy.json
```

Registry mapping:

```text
mode=native:
  puppeteer -> NativePuppeteerOrchestrator
  masrouter -> NativeMASRouterOrchestrator
  gptswarm  -> NativeGPTSwarmOrchestrator

mode=simulated:
  puppeteer -> PuppeteerOrchestrator
  masrouter -> MASRouterOrchestrator
  gptswarm  -> GPTSwarmOrchestrator
```

등록 위치:

```text
src/orchestrators/registry.py
```

새 orchestrator를 추가하려면 다음 순서로 진행한다.

```text
1. BaseOrchestrator를 상속한 adapter class 추가
2. OrchestratorSpec에 repo_dir, repo_url, expected_paths 정의
3. run_inference 구현
4. ORCHESTRATORS 또는 NATIVE_ORCHESTRATORS registry에 name -> class 추가
5. configs/fedpao_six_clients.yaml 또는 새 config의 client.mas에 name 사용
```

## 6. 검증 명령

Orchestrator repo 상태 확인:

```bash
source .venv/bin/activate
python3 scripts/check_orchestrators.py
```

예상 출력:

```text
puppeteer  OK
  repo: third_party/ChatDev-puppeteer
masrouter  OK
  repo: third_party/masrouter
gptswarm   OK
  repo: third_party/GPTSwarm
```

JSON 출력:

```bash
python3 scripts/check_orchestrators.py --json
```

Python syntax 검증:

```bash
python3 -m compileall scripts src
```

CLI 옵션 확인:

```bash
.venv/bin/python scripts/run_client_inference.py --help
```

전체 6-client native inference smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_client_inference.py \
  --max-samples-per-client 1 \
  --max-new-tokens 64 \
  --dtype float16
```

Puppeteer 설정을 명시해서 실행:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_client_inference.py \
  --mas puppeteer \
  --puppeteer-reward-model-path /data/Llama-3.1-Nemotron-70B-Reward-HF/ \
  --puppeteer-global-config configs/orchestrators/puppeteer_global.yaml \
  --puppeteer-policy-config configs/orchestrators/puppeteer_policy.json
```

특정 orchestrator baseline만 실행:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_client_inference.py --mas puppeteer
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_client_inference.py --mas masrouter
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_client_inference.py --mas gptswarm
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_client_inference.py --mas puppeteer gptswarm
```

Simulated mode 확인:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_client_inference.py \
  --orchestrator-mode simulated \
  --max-samples-per-client 1 \
  --max-new-tokens 64 \
  --dtype float16
```

기본 출력:

```text
outputs/client_inference_smoke.jsonl
```

## 7. JSONL 출력 스키마

각 JSONL row의 공통 필드:

```json
{
  "client_id": "c1",
  "mas": "puppeteer",
  "dataset": "mmlu_pro",
  "dataset_name": "TIGER-Lab/MMLU-Pro",
  "split": "validation",
  "example_index": 0,
  "model_path": "/data/digirl/checkpoints/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/",
  "orchestrator_mode": "native",
  "orchestrator_metadata": {},
  "response": "...",
  "prediction": "A",
  "ground_truth": "A",
  "correct": true,
  "latency_sec": 1.2345
}
```

`response`는 `orchestrator_metadata.response`를 row top-level로 복사한 값이다. `prediction`은 dataset별 단순 post-processing 결과다.

- `mmlu_pro`: 응답에서 `A`-`J` 단일 letter를 추출
- `gsm_hard`: 응답에서 마지막 numeric token을 추출

Puppeteer row metadata 예시:

```json
{
  "orchestrator_metadata": {
    "adapter_mode": "native",
    "orchestrator": "puppeteer",
    "orchestrator_display_name": "Puppeteer",
    "orchestrator_repo_path": "third_party/ChatDev-puppeteer",
    "orchestrator_repo_available": true,
    "orchestrator_missing_paths": [],
    "native_entrypoint": "puppeteer.tasks.runner.BenchmarkRunner.run_reasoning",
    "response": "...",
    "puppeteer_reward_model_path": "/data/Llama-3.1-Nemotron-70B-Reward-HF/",
    "puppeteer_global_config": "/home/flagent/workspace/FedPAO/configs/orchestrators/puppeteer_global.yaml",
    "puppeteer_policy_config": "/home/flagent/workspace/FedPAO/configs/orchestrators/puppeteer_policy.json",
    "puppeteer_policy_training": false,
    "puppeteer_policy_loading": false,
    "puppeteer_policy_load_disabled": true,
    "puppeteer_policy_save_disabled": true
  }
}
```

MASRouter row metadata 예시:

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

GPT-Swarm row metadata 예시:

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

Simulated mode metadata 예시:

```json
{
  "orchestrator_metadata": {
    "adapter_mode": "simulated_local",
    "orchestrator": "puppeteer",
    "orchestrator_display_name": "Puppeteer",
    "orchestrator_repo_path": "third_party/ChatDev-puppeteer",
    "orchestrator_repo_available": true,
    "orchestrator_missing_paths": [],
    "response": "..."
  }
}
```

현재 `run_client_inference.py`는 adapter 내부 예외를 잡아서 JSONL error row로 바꾸지 않는다. baseline import, dependency, runtime error가 발생하면 전체 실행이 중단된다.

결과:

```text
baseline    client  dataset   pred  gt          correct  latency_sec  
puppeteer   c1      mmlu_pro  G     A           false    561.5555     
puppeteer   c2      gsm_hard  -1    -9867630.0  false    249.0821     
masrouter   c3      mmlu_pro  A     A           true     22.4878      
masrouter   c4      gsm_hard  16    -9867630.0  false    11.1148      
gptswarm    c5      mmlu_pro  A     A           true     13.4452   
gptswarm    c6      gsm_hard  16    -9867630.0  false    12.2891   
```

## 8. Local Model Backend

`scripts/run_client_inference.py`는 Hugging Face local path를 기준으로 backend를 로드한다.

로드 순서:

```text
1. AutoProcessor + AutoModelForImageTextToText
2. 실패 시 AutoTokenizer + AutoModelForCausalLM
```

모델 경로 우선순위:

```text
1. --model-path
2. configs/fedpao_six_clients.yaml model.model_id
3. DEFAULT_MODEL_PATH in scripts/run_client_inference.py
```

현재 config의 기본 모델은 Qwen2.5-7B-Instruct snapshot이다. `scripts/run_client_inference.py` 내부 `DEFAULT_MODEL_PATH`는 fallback 값으로 LLaVA-OneVision Qwen2 0.5B 경로를 유지한다.

Generation 설정:

```text
max_new_tokens: --max-new-tokens, default 64
do_sample: false
dtype: auto | float16 | bfloat16 | float32
device_map: auto
local_files_only: true
```

## 9. Dependency 상태

FedPAO `requirements.txt`에는 현재 native smoke 실행을 위해 baseline dependency가 누적되어 있다.

Puppeteer 관련:

```text
arxiv, pathvalidate, markdownify, puremagic, mammoth, python_pptx,
pydub, SpeechRecognition, opencv_python_headless, easyocr, pytube,
whisper, diskcache, easydict, pyvis, seaborn, chromadb
```

GPT-Swarm/MASRouter 관련:

```text
loguru, class-registry, shortuuid, groq, sentence-transformers,
tiktoken, astunparse, langchain-community, python-docx, markdown,
PyPDF2, openpyxl, pylatexenc, google-api-python-client
```

주의:

- 세 baseline의 원본 dependency 요구사항은 서로 다르다.
- 현재는 FedPAO `.venv`에서 native smoke를 우선 수행하는 구성이다.
- dependency conflict가 커지면 baseline별 별도 environment 또는 subprocess runner로 분리하는 것이 다음 선택지다.

## 10. 현재 설계 판단

현재 단계의 원칙:

```text
1. 외부 repo는 third_party에 clone하고 FedPAO git에는 포함하지 않는다.
2. FedPAO 내부에는 안정적인 공통 adapter interface를 둔다.
3. run_client_inference.py의 기본 실행은 native adapter를 사용한다.
4. dependency가 부족하면 simulated로 fallback하지 않고 명시적으로 실패시킨다.
5. Puppeteer config source는 FedPAO configs/orchestrators 아래에 둔다.
6. third_party config 파일은 실행 중 compatibility projection 대상으로만 사용한다.
7. dependency-light prompt wrapping 확인이 필요할 때만 --orchestrator-mode simulated를 사용한다.
8. 현재 JSONL은 smoke 결과와 baseline metadata 중심이며, full trajectory schema는 아직 별도 작업이다.
```

## 11. Native Adapter 보강 기준

현재 native adapter는 baseline runner 호출과 local LLM patching에 초점을 둔다. full FedPAO evaluation으로 보강하려면 다음 조건을 만족해야 한다.

공통 조건:

- FedPAO task example을 baseline 입력 schema로 안정적으로 변환한다.
- baseline output을 FedPAO의 `prediction`, `response`, `trajectory` schema로 변환한다.
- local frozen LLM backend, OpenAI-compatible local endpoint, API backend 선택을 config로 제어한다.
- agent call count, tool call count, token count, latency를 수집한다.
- 실패 시 client/example 단위로 error metadata를 남기고 전체 run이 중단되지 않게 한다.

Puppeteer:

- FedPAO config 투영 방식을 유지하되, abnormal termination 후 복구 검사를 자동화한다.
- smoke inference 중 생성되는 Puppeteer checkpoint/log/artifact 경로를 정리하고, 필요하면 전역 저장 차단 또는 별도 output directory 격리를 추가한다.
- reward model call 비용과 policy action trace를 metadata로 남긴다.
- external tool off 상태와 on 상태를 별도 config profile로 분리한다.

MASRouter:

- routing 결과, 선택된 task, reasoning mode, role, model list를 trajectory metadata로 남긴다.
- `task_label`과 `prompt_file` mapping을 config화한다.
- API backend와 local LLM backend를 선택 가능하게 한다.

GPT-Swarm:

- Swarm graph template을 MMLU-Pro/GSM-Hard별 config로 분리한다.
- `edge_optimize` 같은 학습/최적화 플래그를 config에서 제어한다.
- swarm node별 response와 edge traversal 정보를 trajectory metadata로 남긴다.

## 12. Known Caveats

- Native mode는 실제 baseline code path를 호출하지만, 아직 full FedPAO 성능 평가가 아니라 local LLM 연결 smoke test에 가깝다.
- `third_party/`는 git에 포함되지 않으므로 새 환경에서는 clone 명령을 다시 실행해야 한다.
- Codex sandbox 내부에서는 GPU device node나 dataset cache가 보이지 않을 수 있다.
- Puppeteer config projection은 정상 예외 흐름에서는 원복되지만, process kill에는 취약하다.
- 현재 `run_client_inference.py`는 baseline별 runtime error를 row-level error로 저장하지 않고 즉시 중단한다.
- baseline별 trajectory schema 정리가 끝나기 전까지 cost, redundant call, invalid action metric은 완전하게 계산되지 않는다.
