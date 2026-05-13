# Client Interaction Results

검증일: 2026-05-13 KST

이 문서는 현재 구현된 FedPAO scaffold에서 실제로 어떤 입력이 각 client에 들어갔고, client 내부에서 어떤 순서로 상호작용이 일어났는지 실행 결과 기준으로 정리한다.

중요한 전제:

- 현재 `puppeteer`, `masrouter`, `gptswarm`는 실제 외부 MAS repo adapter가 아니라 mock orchestrator 안에서 client label로만 사용된다.
- 현재 client interaction은 `one-step local frozen LLM call`이다.
- 실제 Puppeteer / MASRouter / GPT-Swarm multi-agent loop, routing, graph execution은 아직 연결하지 않았다.
- server-side FedPAO dry run 로그에는 raw question, raw prompt, raw response를 저장하지 않는다. 아래 실제 입력은 같은 config/split/seed로 local loader와 prompt formatter를 다시 호출해 재구성한 것이다.

## 1. 실행 기준

사용 config:

```text
configs/fedpao_six_clients.yaml
```

핵심 설정:

```yaml
model:
  backend: hf_causal
  model_id: /data/digirl/checkpoints/Qwen3.5-2B-Base

prompt:
  latent_dim: 32
  prompt_length: 8
  projection_seed: 42

optimizer:
  sigma0: 0.5
  population_size: 8
  local_iterations: 5

federated:
  aggregation: fedavg
  weight_by: num_examples
```

실제로 검증한 dry run 명령:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python scripts/run_fedpao_dryrun.py \
  --config configs/fedpao_six_clients.yaml \
  --rounds 1 \
  --clients-per-round 6 \
  --max-train-samples-per-client 1 \
  --population-size 2 \
  --local-iterations 1 \
  --max-new-tokens 2 \
  --output-dir outputs/fedpao_dryrun_phase1_seeded
```

이 명령은 성능 실험이 아니라 end-to-end smoke test다.

실제 계산량:

```text
rounds = 1
clients = 6
examples per client = 1
CMA population = 2
CMA local iterations = 1
model calls = 1 * 6 * 1 * 2 * 1 = 12 local LLM calls
```

## 2. Client 내부 상호작용 순서

각 client에서 실제로 일어나는 일은 다음과 같다.

```text
server
  └─ broadcasts z_global
       initial z_global = zeros(32)

client c_k
  ├─ loads local examples from deterministic split
  ├─ receives z_global as CMA-ES initialization point
  ├─ CMA-ES samples candidate latent vectors
  │    population_size = 2 in smoke run
  ├─ for each candidate z:
  │    ├─ SoftPromptProjector maps z -> soft_prompt_embeds
  │    │    shape = (prompt_length, hidden_size) = (8, 2048)
  │    ├─ task prompt is tokenized
  │    ├─ input embeddings are formed as:
  │    │    concat(soft_prompt_embeds, token_embeds)
  │    ├─ frozen Qwen3_5ForCausalLM generates max_new_tokens=2
  │    ├─ MockOrchestrator emits one trajectory step
  │    ├─ LocalClient normalizes prediction
  │    └─ LocalClient computes exact-match accuracy and objective loss
  ├─ chooses local best z_client
  └─ sends only z_client and aggregate metrics to server

server
  └─ FedAvg aggregates 6 client latents into next z_global
```

The current mock trajectory schema for each client call is:

```json
{
  "step_id": 1,
  "selected_component": "llm",
  "capability": "direct_answer",
  "tokens": "<prompt + generated tokens>",
  "latency_sec": "<generation latency>",
  "is_redundant": false
}
```

No tool calls or real agent-to-agent messages occur yet.

## 3. 실제 Dry Run 입력

Dry run은 `split_role=train`을 사용한다. MMLU-Pro는 config상 `train_split=validation`이므로 validation 첫 샘플이 들어갔고, GSM-Hard는 seed 42 split file의 train 첫 인덱스가 들어갔다.

### 3.1 MMLU-Pro Clients: c1, c3, c5

공통 task:

```text
task_id: 06c2a7a664bd
source split: validation
ground truth: A
```

Question:

```text
The symmetric group $S_n$ has $
\factorial{n}$ elements, hence it is not true that $S_{10}$ has 10 elements.
Find the characteristic of the ring 2Z.
```

Options:

```text
A. 0
B. 30
C. 3
D. 10
E. 12
F. 50
G. 2
H. 100
I. 20
J. 5
```

Actual prompt for `c1 / puppeteer`:

```text
You are the local frozen LLM backend for the puppeteer client.
Answer the multiple-choice question. Return only the option letter.

Question: The symmetric group $S_n$ has $
\factorial{n}$ elements, hence it is not true that $S_{10}$ has 10 elements.
Find the characteristic of the ring 2Z.
Options:
A. 0
B. 30
C. 3
D. 10
E. 12
F. 50
G. 2
H. 100
I. 20
J. 5
Answer:
```

`c3 / masrouter` and `c5 / gptswarm` use the same task and options. The only prompt difference is the client label:

```text
You are the local frozen LLM backend for the masrouter client.
```

or:

```text
You are the local frozen LLM backend for the gptswarm client.
```

At this stage this label does not invoke a real MAS policy. It only changes the text prompt and therefore can slightly change token count.

### 3.2 GSM-Hard Clients: c2, c4, c6

공통 task:

```text
task_id: 6be59461ea2f
source split: train
ground truth: 5.9749e-06
```

Problem:

```text
Milo is making a mosaic with chips of glass. It takes 8033625 glass chips to make every square inch of the mosaic. A bag of glass chips holds 72 chips. Milo wants his mosaic to be three inches tall. If he has two bags of glass chips, how many inches long can he make his mosaic?
```

Actual prompt for `c2 / puppeteer`:

```text
You are the local frozen LLM backend for the puppeteer client.
Solve the math problem. Return only the final numeric answer.

Problem: Milo is making a mosaic with chips of glass. It takes 8033625 glass chips to make every square inch of the mosaic. A bag of glass chips holds 72 chips. Milo wants his mosaic to be three inches tall. If he has two bags of glass chips, how many inches long can he make his mosaic?
Answer:
```

`c4 / masrouter` and `c6 / gptswarm` use the same problem. The only prompt difference is again the client label:

```text
You are the local frozen LLM backend for the masrouter client.
```

or:

```text
You are the local frozen LLM backend for the gptswarm client.
```

## 4. 실행 결과: Mock Orchestrator

Mock orchestrator direct check:

```text
outputs/mock_orchestrator_phase1/predictions.jsonl
outputs/mock_orchestrator_phase1/client_metrics.jsonl
```

이 실행은 `split_role=eval`, `max_new_tokens=2`로 수행됐다. 각 client는 soft prompt를 붙인 frozen LM을 한 번 호출했고, 모두 `<think>`를 생성했다.

| client | MAS label | dataset | task_id | final answer | ground truth | correct | prompt tokens | generated tokens | total tokens | latency sec |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| c1 | puppeteer | mmlu_pro | 06c2a7a664bd | `<think>` | A | false | 147 | 2 | 149 | 1.2575 |
| c2 | puppeteer | gsm_hard | 123cd3d2aa06 | `<think>` | 9.0 | false | 71 | 2 | 73 | 0.1216 |
| c3 | masrouter | mmlu_pro | 06c2a7a664bd | `<think>` | A | false | 147 | 2 | 149 | 0.0929 |
| c4 | masrouter | gsm_hard | 123cd3d2aa06 | `<think>` | 9.0 | false | 71 | 2 | 73 | 0.0907 |
| c5 | gptswarm | mmlu_pro | 06c2a7a664bd | `<think>` | A | false | 149 | 2 | 151 | 0.0929 |
| c6 | gptswarm | gsm_hard | 123cd3d2aa06 | `<think>` | 9.0 | false | 73 | 2 | 75 | 0.0907 |

Interpretation:

- Accuracy가 0인 것은 현재 성능 평가가 아니라 generation path smoke test이기 때문이다.
- `max_new_tokens=2`라서 Qwen 계열의 thinking marker인 `<think>`만 나온다.
- token count 차이는 client label 문자열이 다르기 때문이다. `gptswarm` label이 조금 더 길어 MMLU/GSM 모두 token count가 2개 더 컸다.
- `agent_calls=1`, `tool_calls=0`, `redundant_call=0`이다.

## 5. 실행 결과: Single-Client CMA-ES

단일 client smoke:

```text
outputs/local_cma_phase1/metrics.json
outputs/local_cma_phase1/best_latent.npy
```

결과:

```json
{
  "client_id": "c1",
  "initial_loss": 1.199,
  "best_loss": 1.199,
  "num_evaluations": 3,
  "best_z_norm": 0.0
}
```

Objective 계산:

```text
loss =
  task_loss_weight * (1 - accuracy)
  + call_cost_weight * avg_agent_calls
  + token_cost_weight * avg_total_tokens
  + redundant_call_weight * redundant_call_rate
```

For c1:

```text
accuracy = 0.0
avg_agent_calls = 1
avg_total_tokens = 149
redundant_call_rate = 0

loss = 1.0 * (1 - 0)
     + 0.05 * 1
     + 0.001 * 149
     + 0.1 * 0
     = 1.199
```

현재 구현은 initial global latent도 baseline으로 먼저 평가한다. 따라서 이 smoke run에서는 initial latent 1개와 `population_size=2`, `local_iterations=1`의 candidate latent 2개를 함께 평가해 총 3회 평가한다. candidate가 initial보다 개선되지 않으면 best latent는 initial global latent로 유지된다.

## 6. 실행 결과: 6-Client FedPAO Dry Run

Server-side logs:

```text
outputs/fedpao_dryrun_phase1_seeded/client_metrics.jsonl
outputs/fedpao_dryrun_phase1_seeded/metrics.jsonl
outputs/fedpao_dryrun_phase1_seeded/round_000/global_latent_input.npy
outputs/fedpao_dryrun_phase1_seeded/round_000/global_latent_output.npy
```

Client metrics:

| round | client | MAS label | examples | local loss | local accuracy | z norm | upload bytes | evaluations |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | c1 | puppeteer | 1 | 1.0 | 0.0 | 0.0000 | 128 | 3 |
| 0 | c2 | puppeteer | 1 | 1.0 | 0.0 | 0.0000 | 128 | 3 |
| 0 | c3 | masrouter | 1 | 1.0 | 0.0 | 0.0000 | 128 | 3 |
| 0 | c4 | masrouter | 1 | 1.0 | 0.0 | 0.0000 | 128 | 3 |
| 0 | c5 | gptswarm | 1 | 1.0 | 0.0 | 0.0000 | 128 | 3 |
| 0 | c6 | gptswarm | 1 | 1.0 | 0.0 | 0.0000 | 128 | 3 |

Round-level metrics:

```json
{
  "round": 0,
  "num_clients": 6,
  "global_z_norm": 0.0,
  "client_upload_bytes": 768,
  "server_download_bytes": 768,
  "total_round_bytes": 1536
}
```

Communication calculation:

```text
latent_dim = 32
dtype = float32
bytes per latent = 32 * 4 = 128
clients = 6
client upload bytes = 128 * 6 = 768
server download bytes = 128 * 6 = 768
total round bytes = 1536
```

The FedAvg output is saved at:

```text
outputs/fedpao_dryrun_phase1_seeded/round_000/global_latent_output.npy
```

## 7. 각 “Agent System”별 현재 의미

현재 구현에서 세 agent system은 실제 외부 framework가 아니라 동일한 mock orchestrator를 공유한다.

| MAS label | clients | 현재 동작 | 실제 구현 상태 |
| --- | --- | --- | --- |
| Puppeteer | c1, c2 | prompt에 `puppeteer client` label 삽입 후 direct LLM call | ChatDev Puppeteer adapter 미연결 |
| MASRouter | c3, c4 | prompt에 `masrouter client` label 삽입 후 direct LLM call | MASRouter adapter 미연결 |
| GPT-Swarm | c5, c6 | prompt에 `gptswarm client` label 삽입 후 direct LLM call | GPTSwarm adapter 미연결 |

따라서 현재 상호작용 차이는 다음 두 가지뿐이다.

```text
1. dataset 차이:
   MMLU-Pro: option-letter answer
   GSM-Hard: final numeric answer

2. MAS label text 차이:
   puppeteer / masrouter / gptswarm 문자열이 prompt에 들어감
```

아직 없는 것:

```text
- multi-agent role communication
- tool execution
- routing decision
- graph node selection
- redundant call detection from real trajectory
- MAS-specific planner/executor/controller interaction
```

## 8. Privacy / Logging 관점

현재 dry run server log에 남는 값:

```text
round
client_id
num_examples
local_loss
local_accuracy
z_norm
communication_bytes
client_upload_bytes
server_download_bytes
total_round_bytes
num_evaluations
global_z_norm
```

현재 dry run server log에 남기지 않는 값:

```text
raw question
raw prompt
raw model response
full trajectory text
local candidate latent list
```

이 구조는 FedPAO proposal의 기본 의도와 맞다. 즉, server는 raw execution trace를 모으지 않고 low-dimensional latent와 aggregate metric만 받는다.

## 9. 다음에 실제 MAS를 붙이면 바뀌는 부분

현재 mock orchestrator:

```text
TaskExample + z
  -> one prompt
  -> one frozen LLM call
  -> one trajectory step
```

실제 MAS adapter 연결 후 목표:

```text
TaskExample + z
  -> MAS-specific planner/router/graph controller
  -> multiple component or agent calls
  -> common trajectory schema
  -> final answer
  -> same evaluator/objective
```

공통 schema는 유지해야 한다.

```json
{
  "step_id": 1,
  "selected_component": "math_agent",
  "capability": "calculation",
  "tokens": 512,
  "latency_sec": 0.8,
  "is_redundant": false
}
```

즉, 다음 구현에서 바뀌어야 할 것은 `MockOrchestrator.run()` 내부뿐이고, `LocalClient`, `CMA-ES`, `FedAvg`는 그대로 재사용하는 것이 목표다.
