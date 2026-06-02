# `scripts/run.py` Round 학습 흐름

이 문서는 `scripts/run.py`가 `round_000`을 먼저 학습한 뒤, 이후 round에서 그 결과물을 어떻게 다시 사용해 학습을 이어가는지 정리한다.

## 전체 구조

`scripts/run.py`의 실행 단위는 하나의 run directory다. 기본 위치는 `outputs/runs/<run-name>/`이며, 각 round는 아래처럼 독립 directory를 가진다.

```text
outputs/runs/<run-name>/
  config.yaml
  round_000/
    manifest.json
    <client_id>_<mas>_<dataset>/
      .hydra/
      checkpoints/
      manifest.json
  round_001/
    inference.jsonl
    analysis_manifest.json
    manifest.json
    server/
      global_prompt_manifest.json
      global_prompt_round002.pt
    <client_id>_<mas>_<dataset>/
      trajectories.jsonl
      analysis_report.json
      analysis_report.md
      prompt_round001.pt
      prompt_manifest.json
      checkpoints/
      manifest.json
```

실행 흐름은 크게 두 단계다.

1. `round0_train()`이 client별 초기 MAS checkpoint를 만든다.
2. `run_federated_round()`가 이전 round의 checkpoint와, 가능하면 이전 server global prompt를 사용해서 다음 round를 실행한다.

`main()`에서는 `--start-round`에 따라 새 run인지 resume인지 결정한다. `--start-round 0`이면 round0부터 새로 학습하고, `--start-round 1`이면 기존 `round_000/manifest.json`을 읽어 round1부터 시작한다. `--start-round N`이면 `round_(N-1)/manifest.json`을 읽어서 이어서 실행한다.

## Round 0: 초기 client별 학습

Round 0은 `round0_train()`에서 수행된다. 목적은 각 client의 MAS 구현이 이후 round에서 로드할 수 있는 checkpoint를 만드는 것이다.

### 데이터 분할

Round 0에서 각 client는 `split_client_examples()`로 dataset을 섞고 train/validation으로 나눈다.

- seed는 `config["seed"]`와 dataset/client key를 섞어서 만든다.
- `data.train_fraction`이 train 비율을 결정한다.
- `data.max_samples_per_client`가 있으면 전체 sample 수를 먼저 제한한다.
- `--max-train-samples-per-client`, `--max-valid-samples-per-client`가 있으면 train/valid 개수를 추가로 제한한다.

현재 `configs/fedpao_six_clients.yaml` 기준으로는 `train_fraction: 0.8`, `max_samples_per_client: 100`이므로 client별 최대 100개를 뽑고 그중 80%를 round0 train에 사용한다.

Round 1 이후 trajectory 분석용 inference는 `split_client_examples_with_trajectory()`를 사용한다. 이 함수는 같은 seed shuffle을 공유하지만, 먼저 train/valid pool을 만든 뒤 그 pool에 들어가지 않은 held-out index에서 summary agent용 inference 예제를 뽑는다. 따라서 summary agent가 보는 trajectory는 train과 validation에 사용된 데이터와 겹치지 않는다.

현재 config 기준으로는 shuffle된 앞 100개가 train/valid pool이고, `data.max_trajectory_samples_per_client: 3`에 따라 그 뒤 held-out pool에서 3개가 선택된다. `--max-samples-per-client`를 넘기면 이 config 값을 실행 시점에 override한다. 이 held-out split 이름은 manifest에 `trajectory_split`, 개수는 `num_trajectory_examples`로 기록된다.

### trainer 선택

`--round0-trainer` 값에 따라 round0 학습 방식이 갈린다.

- `native`: third_party MAS trainer를 실제로 호출한다. 기본값이다.
- `artifact`: 가벼운 placeholder checkpoint만 만든다. smoke test나 artifact contract 확인용이다.

`native`일 때는 `load_backend()`로 frozen local LLM backend를 한 번 로드하고, 이 backend를 `llm_generate` 함수로 감싼 뒤 각 MAS trainer에 주입한다. 즉 third_party 코드가 OpenAI나 외부 LLM을 직접 호출하는 대신 FedPAO의 local model generate 함수를 호출하도록 patch한다.

### MAS별 round0 checkpoint

`train_client_native()`가 `client["mas"]`에 따라 실제 trainer를 dispatch한다.

- GPT-Swarm: `train_gptswarm_native()`
  - `Swarm(["IO", "COT", "IO"], domain, edge_optimize=True)`를 만든다.
  - batch마다 graph를 sample하고 utility 기반 loss로 `connection_dist`를 업데이트한다.
  - `connection_dist_state_dict`와 `edge_probs`를 `gptswarm_edges_round0.pt`에 저장한다.

- MASRouter: `train_masrouter_native()`
  - `MasRouter(max_agent=3)`를 만든다.
  - task classification loss, answer utility loss, VAE loss를 합쳐 업데이트한다.
  - `router_state_dict`를 `masrouter_round0.pt`에 저장한다.

각 client 학습이 끝나면 client directory에 `.hydra` config, checkpoint, TensorBoard log, `manifest.json`을 저장한다. 마지막으로 `round_000/manifest.json`에는 모든 client record가 모인다.

중요한 것은 이후 round가 model object를 메모리에 들고 이어지는 것이 아니라, `round_000/manifest.json`에 기록된 `checkpoint_path`를 기준으로 다시 로드한다는 점이다.

## Round 1 이후: 이전 round 결과를 사용한 반복 학습

Round 1부터는 `run_federated_round()`가 실행된다. 이 함수의 stage 이름은 manifest에 `infer_analyze_prompt_train_aggregate`로 기록된다.

전체 순서는 다음과 같다.

```text
previous_manifest 로드
  -> previous client checkpoints 로드
  -> previous global prompt 로드, 있으면 LLM 호출에 soft prompt 주입
  -> client별 held-out inference 수행
  -> trajectory 저장
  -> summary agent가 trajectory 분석
  -> 분석문을 soft prompt latent로 변환 및 최적화
  -> client별 prompt artifact 생성
  -> prompt를 LLM embedding 앞에 붙여 client MAS 재학습
  -> client prompt들을 FedAvg로 집계
  -> 다음 round용 global prompt 저장
```

### 1. 이전 round manifest와 checkpoint 로드

`run_federated_round()`는 먼저 `load_round0_models(previous_manifest)`를 호출한다. 함수 이름은 round0로 되어 있지만 실제로는 `previous_manifest["clients"]`에 들어 있는 모든 client checkpoint를 로드한다.

로드 결과는 client id별로 아래 구조를 가진다.

```python
{
    "record": previous_client_manifest_record,
    "state": torch_loaded_checkpoint,
}
```

이전 round가 round0이면 round0 checkpoint를 읽고, 이전 round가 round1 이상이면 직전 round에서 재학습한 checkpoint를 읽는다.

### 2. 이전 global prompt 사용

Round 1은 기본적으로 아직 이전 server prompt가 없으므로 `global_prompt_state`가 `None`일 수 있다. 이 경우 일반 `generate()`를 사용한다.

Round 2부터는 직전 round manifest의 `global_prompt.global_prompt_artifact_path`가 `previous_global_prompt_path`로 넘어온다. 그러면 `load_global_prompt_state()`로 server global prompt를 읽고, LLM 호출은 `generate_with_embedding_soft_prompt()`로 바뀐다.

이 함수는 causal LM backend에서 다음 방식으로 soft prompt를 주입한다.

1. 입력 prompt를 tokenizer로 tokenizing한다.
2. token id를 embedding으로 변환한다.
3. global prompt의 `soft_prompt` latent를 `latent_to_virtual_prompt_embeddings()`로 virtual token embedding으로 투영한다.
4. virtual token embedding을 원래 token embedding 앞에 concatenate한다.
5. `inputs_embeds`와 확장된 attention mask로 `model.generate()`를 호출한다.

따라서 이후 round의 LLM 호출은 text prompt 자체가 바뀌는 것이 아니라, embedding level prefix가 붙은 상태로 실행된다.

### 3. 이전 checkpoint를 넣은 MAS inference

각 client에 대해 `create_orchestrator()`를 호출할 때 `round0_checkpoint_path=record["checkpoint_path"]`를 넘긴다. 변수 이름은 round0이지만, 여기서 `record`는 직전 round manifest의 client record다.

MAS별 native orchestrator는 이 경로를 다음처럼 사용한다.

- Puppeteer
  - policy config에 `training: false`, `loading: true`를 설정한다.
  - `paths.model_path`에 checkpoint path를 넣는다.
  - inference 중에는 save를 비활성화한다.

- GPT-Swarm
  - checkpoint의 `connection_dist_state_dict`를 `swarm.connection_dist`에 로드한다.
  - edge probability가 0.5보다 큰 edge mask로 graph를 realize한다.

- MASRouter
  - checkpoint의 `router_state_dict`를 `MasRouter`에 로드한다.
  - 로드된 router로 `router.forward()` inference를 수행한다.

이 inference 결과는 `round_<N>/inference.jsonl`과 client별 `trajectories.jsonl`에 저장된다. 각 row에는 prediction, ground truth, correctness, latency, native metadata, 이전 checkpoint path, global prompt 사용 여부가 들어간다.

### 4. trajectory 분석

client별 inference가 끝나면 `write_client_analysis_report()`가 실행된다.

이 함수는 `build_summary_agent_prompt()`로 trajectory observations를 만들고, frozen local LLM을 summary agent처럼 호출한다. summary agent는 단순한 정확도 요약이 아니라 다음 round routing에 도움이 되는 패턴을 최대 3문장으로 만들도록 지시받는다.

결과는 아래 파일에 저장된다.

- `analysis_report.json`
- `analysis_report.md`

### 5. 분석 report에서 client prompt artifact 생성

`write_client_prompt_artifact()`는 analysis summary를 soft prompt로 바꾼다.

구현 단계는 다음과 같다.

1. `sentence_embedding_from_report()`
   - summary text를 frozen LM input embedding으로 mean pooling한다.

2. `project_report_embedding_to_latent()`
   - report embedding을 `prompt.latent_dim` 차원의 latent로 random projection한다.

3. `gradient_free_refine_prompt_latent()`
   - `optimizer.name`이 `cma_es`, `cma-es`, `cma` 중 하나이면 `cma.CMAEvolutionStrategy`를 사용한다.
   - 시작점은 report summary에서 만든 `report_latent`다.
   - CMA-ES는 매 iteration마다 `population_size`개 후보 latent를 `ask()`로 생성한다.
   - 각 후보는 `score = cosine(report_latent, candidate) + accuracy - cost penalties`로 평가된다.
   - `cma` 라이브러리는 minimize API이므로 코드에서는 `loss = -score`를 `tell()`에 넘긴다.
   - `tell()` 이후 CMA-ES 내부의 평균과 공분산이 업데이트되고, 다음 generation 후보 분포가 바뀐다.
   - 가장 점수가 좋은 latent를 client soft prompt로 선택한다.

4. `prompt_round<N>.pt`와 `prompt_manifest.json` 저장
   - `soft_prompt`
   - `report_embedding`
   - `report_latent`
   - `bridge_text`
   - optimization history
   - accuracy metadata

이 client prompt는 바로 다음 단계의 local 재학습에 사용된다.

### CMA-ES hyperparameter 의미

현재 config의 관련 값은 아래와 같다.

```yaml
prompt:
  type: soft_latent
  latent_dim: 64
  init_std: 0.02

optimizer:
  name: cma_es
  sigma0: 0.5
  population_size: 12
  local_iterations: 10
```

- `latent_dim`
  - CMA-ES가 탐색하는 prompt latent 차원이다.
  - report embedding은 이 차원으로 random projection된다.
  - client prompt와 server global prompt의 `soft_prompt` 크기도 이 값에 의해 결정된다.
  - 차원이 커질수록 표현력은 늘지만 CMA-ES 탐색 비용과 필요한 population이 증가한다.

- `intrinsic_dim`
  - 현재 `configs/fedpao_six_clients.yaml`에는 설정되어 있지 않다.
  - 현재 `scripts/run.py`에서도 사용되지 않는다.
  - 일반적으로는 CMA-ES가 탐색하는 더 낮은 intrinsic vector 차원과 실제 prompt latent 차원을 분리할 때 쓰는 이름이다.
  - 현재 구현은 `CMA-ES candidate dimension == latent_dim` 구조이므로 `intrinsic_dim`은 문서적 placeholder에 가깝다.

- `init_std`
  - report latent와 soft prompt scale을 작게 유지하는 값이다.
  - embedding prefix가 너무 커지면 원래 prompt semantics를 강하게 밀어낼 수 있어서 작은 초기 scale을 둔다.

- `sigma0`
  - CMA-ES 초기 탐색 반경이다.
  - 코드에서는 실제 sigma가 `sigma0 * init_std`로 들어간다.
  - 예를 들어 `sigma0: 0.5`, `init_std: 0.02`이면 초기 표준편차는 `0.01`이다.

- `population_size`
  - CMA-ES 한 generation에서 평가하는 후보 수다.
  - 값이 클수록 분포 업데이트가 안정적이지만 client 수와 round 수에 비례해 비용이 늘어난다.

- `local_iterations`
  - CMA-ES generation 수다.
  - 총 후보 평가는 대략 `population_size * local_iterations`개다.
  - `population_size: 12`, `local_iterations: 10`이면 client별 120개 후보를 proxy objective로 평가한다.

현재 CMA-ES objective는 후보마다 실제 MAS inference를 다시 돌리는 비싼 objective가 아니라, summary report latent와 candidate latent의 alignment 및 cost penalty를 보는 proxy objective다. 따라서 round loop 안에서도 비교적 낮은 비용으로 client별 prompt latent를 조정할 수 있다.

### 6. prompt를 주입한 client 재학습

`train_client_with_prompt()`는 방금 만든 `prompt_round<N>.pt`를 `load_global_prompt_state()`로 읽어 `prompt_state`로 만든 뒤 학습한다.

여기서 중요한 차이는 `llm_generate`가 일반 generate가 아니라 `generate_with_embedding_soft_prompt()`라는 점이다. 즉 third_party MAS trainer는 기존 round0과 같은 native training loop를 돌지만, trainer 내부에서 LLM을 호출할 때는 client analysis로부터 만든 soft prompt embedding이 모든 prompt 앞에 붙는다.

`--round0-trainer native`이면 다시 `train_client_native()`를 호출한다. 그래서 round1 이후에도 Puppeteer, GPT-Swarm, MASRouter 각각의 native train loop가 실행되고, 새 checkpoint가 client directory의 `checkpoints/` 아래에 저장된다.

이 새 checkpoint path는 client manifest에 기록되고, 다음 round의 `previous_manifest`로 사용된다.

### 7. FedAvg로 server global prompt 생성

모든 client prompt가 만들어지면 `fedavg_prompt_artifacts()`가 실행된다.

이 함수는 각 client의 `prompt_round<N>.pt`에서 `soft_prompt` tensor를 읽고 weighted average를 계산한다.

- weight 기준은 `federated.weight_by`이며 기본 config에서는 `num_examples`다.
- aggregation 이름은 `federated.aggregation`이며 기본 config에서는 `fedavg`다.
- latent dimension이 서로 다르면 에러를 낸다.

결과는 server artifact로 저장된다.

```text
round_<N>/server/global_prompt_round<N+1>.pt
round_<N>/server/global_prompt_manifest.json
```

이 global prompt는 round `N+1`에서 `previous_global_prompt_path`로 전달되어 모든 client inference와 summary 이후 학습의 LLM 호출에 사용된다.

## Resume 동작

`--start-round`를 사용하면 이전 artifact를 재사용한다.

- `--start-round 1`
  - `round_000/manifest.json`을 읽는다.
  - round0 학습을 건너뛰고 round1부터 실행한다.

- `--start-round 2`
  - `round_001/manifest.json`을 읽는다.
  - `round_001/server/global_prompt_round002.pt`를 찾아 round2에 사용한다.

- `--start-round N`
  - `round_(N-1)/manifest.json`을 읽고 round N부터 실행한다.

manifest 안의 path가 기존 run root 이름을 포함하고 있는데 현재 위치와 맞지 않으면 `rebase_round0_manifest_paths()`가 `outputs/runs/<run-name>/...` 뒤의 상대 경로를 현재 `run_root` 기준으로 다시 맞춘다.

## Round 간 전달되는 핵심 artifact

Round 간에는 크게 두 종류가 전달된다.

첫 번째는 client MAS checkpoint다.

```text
previous round client manifest
  -> checkpoint_path
  -> create_orchestrator(..., round0_checkpoint_path=checkpoint_path)
  -> native MAS inference에 로드
  -> train_client_with_prompt()가 새 checkpoint 생성
  -> current round manifest의 checkpoint_path로 기록
```

두 번째는 server global soft prompt다.

```text
current round client analysis
  -> client prompt_round<N>.pt
  -> FedAvg
  -> server/global_prompt_round<N+1>.pt
  -> next round generate_with_embedding_soft_prompt()
```

즉 checkpoint는 MAS 내부 routing/policy/edge/router state를 이어주고, global soft prompt는 frozen LLM 호출 앞단의 embedding prefix로 이어진다.

## 현재 구현상 주의할 점

- `round0_checkpoint_path`라는 인자명은 round1 이후에도 계속 쓰이지만, 실제 의미는 "직전 round의 client checkpoint path"다.
- `load_round0_models()`도 이름은 round0지만 `previous_manifest`의 client checkpoint를 일반적으로 로드한다.
- 실행할 round 범위는 config가 아니라 `scripts/run.py --rounds`와 `--start-round`가 제어한다.
- 현재 `run_federated_round()`는 별도 client sampling 없이 선택된 모든 client를 매 round 순회한다.
- `federated.weight_by: num_examples`는 prompt manifest의 `num_examples`를 기준으로 FedAvg weight를 준다.
- embedding-level soft prompt injection은 `causal_lm` backend만 지원한다. image-text backend에서는 현재 에러를 낸다.
- `artifact` trainer는 실제 MAS 학습이 아니라 checkpoint 계약과 logging을 확인하기 위한 placeholder다.

## 간단 실행 예시

Round0 후 round4까지 실행:

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/run.py \
  --start-round 0 \                  
  --rounds 4
```
