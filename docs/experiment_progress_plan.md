# FedPAO 진행 상태와 다음 실험 계획

검증일: 2026-05-13 KST

이 문서는 1~7번 진행 단계의 현재 완료 상태, 실행 결과, 그리고 다음 실험 전환 계획을 정리한다. 현재 결과는 성능 주장용 결과가 아니라, FedPAO 최소 루프를 실제 실험 규모로 키우기 전의 scaffold 검증 결과다.

## 1. 완료된 진행 단계

### 1. MMLU-Pro choice log-likelihood scoring

완료.

`HFCausalBackend.score_continuations()`를 추가했고, MMLU-Pro client는 생성 문자열을 parsing하는 대신 선택지 `A-J` continuation의 log-likelihood를 계산할 수 있다.

현재 경로:

```text
LocalClient.evaluate_loss()
  -> MockOrchestrator.score_mmlu_choices()
  -> HFCausalBackend.score_choices_with_soft_prompt()
  -> HFCausalBackend.score_continuations()
```

의미:

```text
prediction = argmax log P(option | question, soft_prompt)
task_loss = -log P(correct option | question, soft_prompt)
```

이 방식은 Qwen Base 계열이 `Return only A` 같은 instruction을 항상 따르지 못하는 문제를 줄이고, CMA-ES가 exact-match보다 덜 sparse한 objective를 보도록 한다.

### 2. objective score_mode 분리

완료.

Config에 `objective.score_mode`를 추가했다.

현재 모드:

```text
exact_match
  기존 generation + exact match objective

hybrid_logprob
  MMLU-Pro: option log-likelihood
  GSM-Hard: answer continuation log-likelihood
```

추가 config:

```text
configs/fedpao_logprob_smoke.yaml
```

기존 smoke config는 task-only exact-match 기준으로 유지했다.

```text
configs/fedpao_smoke_taskonly.yaml
```

### 3. 6-client baseline 확장 실행

완료.

실행 명령:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/run_latent_baselines.py \
  --config configs/fedpao_smoke_taskonly.yaml \
  --clients-per-round 6 \
  --max-train-samples-per-client 4 \
  --max-eval-samples-per-client 4 \
  --population-size 4 \
  --local-iterations 2 \
  --max-new-tokens 8 \
  --include-local-cma \
  --output-dir outputs/latent_baselines_6client_s4
```

검증 결과:

```text
baseline_metrics.jsonl rows: 24
6 clients x 4 baselines = 24
local_only_cma num_evaluations: 9
```

Baseline 종류:

```text
no_soft_prompt
zero_latent
random_latent
local_only_cma
```

현재 accuracy는 대부분 0.0이며, 이는 mock/raw prompt와 작은 sample 수 기준의 scaffold sanity check로만 해석한다.

### 4. dense scoring FedPAO dry run 재실행

완료.

실행 명령:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/run_fedpao_dryrun.py \
  --config configs/fedpao_logprob_smoke.yaml \
  --rounds 2 \
  --clients-per-round 6 \
  --max-train-samples-per-client 4 \
  --max-eval-samples-per-client 4 \
  --population-size 4 \
  --local-iterations 2 \
  --max-new-tokens 8 \
  --eval-every-round \
  --output-dir outputs/fedpao_logprob_dryrun_r2_p4_i2_s4
```

검증 결과:

```text
client_metrics.jsonl rows: 12
metrics.jsonl rows: 2
eval_metrics.jsonl rows: 12
num_evaluations: 9
finite_loss: True
round_000/global_latent_output.npy == round_001/global_latent_input.npy
```

Round communication metric:

```text
client_upload_bytes: 768
server_download_bytes: 768
total_round_bytes: 1536
```

Dry run 출력 요약:

```text
round=0 client=c1 loss=4.0508 acc=0.250
round=0 client=c2 loss=24.5342 acc=0.000
round=0 client=c3 loss=4.2207 acc=0.250
round=0 client=c4 loss=24.3779 acc=0.000
round=0 client=c5 loss=4.3281 acc=0.000
round=0 client=c6 loss=24.5244 acc=0.000
round=1 client=c1 loss=4.2461 acc=0.000
round=1 client=c2 loss=24.7275 acc=0.000
round=1 client=c3 loss=3.9590 acc=0.250
round=1 client=c4 loss=24.9082 acc=0.000
round=1 client=c5 loss=3.8398 acc=0.250
round=1 client=c6 loss=24.6250 acc=0.000
```

### 5. GSM-Hard dense scoring 보강

완료.

GSM-Hard는 같은 숫자 정답이 여러 textual form으로 표현될 수 있으므로, 정답 continuation 하나만 점수화하면 objective가 불필요하게 불안정해진다. 이를 줄이기 위해 numeric answer variants를 추가했다.

예:

```text
5.9749e-06 -> ["5.9749e-06", "0.0000059749"]
15.0       -> ["15.0", "15"]
```

현재 경로:

```text
MockOrchestrator.score_answer()
  -> numeric_answer_variants()
  -> HFCausalBackend.score_continuations()
  -> max variant log-prob
```

검증:

```text
direct assertion: gsm_normalization_ok
2-client backend smoke: passed
```

2-client smoke에서 첫 GSM client loss는 단일 표현 기준 약 `30.0234`에서 bare/spaced variant scoring 기준 약 `23.3730`으로 낮아져, variant max-logprob 경로가 실제 적용됨을 확인했다.

### 6. MAS adapter 시작

완료.

실제 외부 MAS runtime은 아직 연결하지 않았다. 대신 다음 최소 scaffold를 추가했다.

```text
src/orchestrators/adapters.py
src/orchestrators/registry.py
scripts/check_mas_adapters.py
```

현재 adapter 상태 확인:

```bash
.venv/bin/python scripts/check_mas_adapters.py --config configs/fedpao_logprob_smoke.yaml
```

출력:

```text
gptswarm   OK       third_party/GPTSwarm
  note: repo present; runtime integration is not enabled yet
masrouter  OK       third_party/masrouter
  note: repo present; runtime integration is not enabled yet
puppeteer  OK       third_party/ChatDev-puppeteer
  note: repo present; runtime integration is not enabled yet
```

의미:

```text
외부 repo 경로와 핵심 디렉토리는 존재한다.
dependency install/runtime import/trajectory extraction은 아직 시작하지 않았다.
FedPAO dry run은 계속 MockOrchestrator를 사용한다.
```

### 7. 실험 전환 계획

완료.

현재는 다음 조건을 만족한다.

```text
1. frozen causal LM
2. continuous soft prompt latent
3. CMA-ES local update
4. FedAvg aggregation
5. exact-match objective와 log-prob objective 분리
6. 6-client baseline artifact 생성
7. 6-client dense scoring dry run artifact 생성
8. 외부 MAS repo status check 가능
```

따라서 다음 단계부터는 “scaffold 구현”에서 “실험 규모 확장 및 adapter 연결”로 넘어갈 수 있다.

## 2. 현재 주의사항

### pytest

현재 `.venv`에는 `pytest` 모듈이 없다.

```text
.venv/bin/python -m pytest tests/test_normalization.py
-> No module named pytest
```

대신 이번 변경은 `py_compile`과 직접 assertion으로 검증했다. 이후 test suite를 공식 검증 루프에 넣으려면 `pytest`를 requirements에 추가하거나 별도 dev requirements를 만드는 것이 좋다.
`requirements.txt`에는 `pytest>=8.0`을 추가했으며, 현재 venv에는 아직 재설치하지 않았다.

### GPU 0 점유

이번 실행은 GPU 1에서 수행했다. GPU 0은 별도 `python3` 프로세스가 약 46GB를 점유하고 있어 log-prob smoke run에서 OOM이 발생했다.

권장:

```bash
CUDA_VISIBLE_DEVICES=1 ...
```

또는 GPU 0 프로세스의 소유/목적을 확인한 뒤 정리한다.

### 결과 해석

현재 결과는 FedPAO 성능 결과가 아니다.

```text
현재 검증한 것:
  algorithm loop, artifact schema, metric logging, dense objective path

아직 검증하지 않은 것:
  실제 GPTSwarm/MASRouter/Puppeteer orchestration
  agent/tool call 감소
  heterogeneous MAS transfer
  full benchmark accuracy improvement
```

## 3. 다음 작업 순서

### Phase 1. 실험 안정성 정리

```text
1. pytest/dev dependency 정리
2. log-prob dry run을 seed 2~3개로 반복
3. output validator script 추가
4. baseline vs FedAvg result summary script 추가
```

권장 명령:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/run_fedpao_dryrun.py \
  --config configs/fedpao_logprob_smoke.yaml \
  --rounds 2 \
  --clients-per-round 6 \
  --max-train-samples-per-client 8 \
  --max-eval-samples-per-client 8 \
  --population-size 4 \
  --local-iterations 2 \
  --max-new-tokens 8 \
  --eval-every-round \
  --output-dir outputs/fedpao_logprob_dryrun_r2_p4_i2_s8_seed42
```

### Phase 2. 본실험 전 mock benchmark

```text
1. rounds 5
2. clients 6
3. train/eval samples 16~32
4. population_size 8
5. local_iterations 3~5
6. zero/random/local-only/FedAvg 비교
```

이 단계까지도 실제 MAS claim은 하지 않는다. 목적은 FedPAO algorithm이 반복 실행 가능하고 metric curve가 안정적으로 저장되는지 확인하는 것이다.

### Phase 3. GPTSwarm adapter

첫 실제 MAS adapter는 GPTSwarm부터 시작한다.

해야 할 일:

```text
1. third_party/GPTSwarm import 방식 확인
2. 별도 dependency 격리 필요 여부 판단
3. TaskExample -> GPTSwarm input 변환
4. GPTSwarm output -> OrchestratorResult 변환
5. trajectory step schema 매핑
6. token/latency/agent call metric 추출
```

완료 기준:

```text
1. gptswarm client 1개가 MMLU-Pro sample 1개를 처리
2. final_answer와 trajectory가 공통 schema로 반환
3. raw trajectory는 client artifact에만 저장
4. server metric에는 latent와 aggregate metric만 저장
```

### Phase 4. MASRouter, Puppeteer adapter

GPTSwarm adapter가 안정화된 뒤 다음 순서로 진행한다.

```text
1. MASRouter
2. Puppeteer
```

Puppeteer는 dependency conflict 가능성이 크므로 마지막에 붙인다.

## 4. 현재 결론

FedPAO scaffold는 이제 `exact_match`와 `hybrid_logprob` 두 objective 모두에서 6-client dry run이 가능하다. 다음 실험은 full benchmark가 아니라, seed 반복과 sample 수 확장을 통해 mock FedPAO loop의 안정성을 먼저 확인한 뒤 GPTSwarm adapter를 붙이는 순서로 진행한다.
