# Native MAS Adapter Integration

검증일: 2026-05-18 KST

이 문서는 `feat/six-client-local-llm-inference` worktree에서 선별적으로 가져온 native MAS adapter 통합 내용을 정리한다. 기존 FedPAO dry-run 브랜치의 soft prompt, CMA-ES, FedAvg 루프를 mainline으로 유지하고, feature 브랜치에서는 실제 Puppeteer / MASRouter / GPT-Swarm native code path 연결 전략만 가져왔다.

## 가져온 내용

추가한 파일:

```text
src/orchestrators/native_baselines.py
scripts/check_orchestrators.py
scripts/run_native_orchestrator_soft_prompt_smoke.py
configs/orchestrators/puppeteer_global.yaml
configs/orchestrators/puppeteer_policy.json
```

수정한 파일:

```text
src/orchestrators/base.py
src/orchestrators/registry.py
scripts/run_fedpao_dryrun.py
configs/fedpao_six_clients.yaml
configs/fedpao_smoke_taskonly.yaml
configs/fedpao_logprob_smoke.yaml
```

가져오지 않은 것:

```text
feature 브랜치의 requirements.txt 전체
feature 브랜치의 run_client_inference.py 전체
feature 브랜치의 configs/fedpao_six_clients.yaml 전체
Qwen2.5-7B-Instruct default 강제 변경
native mode default 강제 변경
```

## 현재 구조

기본 dry run은 여전히 mock adapter를 사용한다.

```yaml
orchestrators:
  default: mock
```

실제 MAS adapter를 쓰려면 CLI에서 명시한다.

```bash
PYTHONPATH=. .venv/bin/python scripts/run_fedpao_dryrun.py \
  --orchestrator-mode native \
  --mas gptswarm \
  --rounds 1 \
  --clients-per-round 1 \
  --max-train-samples-per-client 1 \
  --population-size 2 \
  --local-iterations 1 \
  --max-new-tokens 8
```

## FedPAO loop 연결 방식

기존 mock path:

```text
CMA-ES candidate z
-> MockOrchestrator.run(example, z)
-> backend.generate_with_soft_prompt(...)
-> LocalClient score
```

native path:

```text
CMA-ES candidate z
-> Native*Orchestrator.run(example, z)
-> native MAS 내부 LLM call
-> patched llm_generate(prompt)
-> backend.generate_with_soft_prompt(prompt, projector(z))
-> native MAS final response
-> LocalClient score
```

즉, native adapter 내부 LLM 호출도 현재 candidate latent `z`를 사용한다.

## Adapter별 entrypoint

```text
Puppeteer:
  puppeteer.tasks.runner.BenchmarkRunner.run_reasoning

MASRouter:
  MAR.MasRouter.mas_router.MasRouter.forward

GPT-Swarm:
  swarm.graph.swarm.Swarm.run
```

## 점검 명령

Repo path와 필수 디렉토리 확인:

```bash
.venv/bin/python scripts/check_orchestrators.py --config configs/fedpao_six_clients.yaml
```

Native MAS + soft prompt 단일 example smoke:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python \
  scripts/run_native_orchestrator_soft_prompt_smoke.py \
  --config configs/fedpao_six_clients.yaml \
  --mas gptswarm \
  --split-role train \
  --max-new-tokens 8 \
  --latent random
```

이 script는 adapter 실행 실패도 JSONL error row로 저장한다.

## 주의사항

- native mode는 여전히 smoke/integration 단계다.
- baseline별 dependency가 현재 `.venv`에 모두 설치되어 있다고 가정하지 않는다.
- Puppeteer는 third_party config에 FedPAO config를 runtime projection한 뒤 정상 종료 시 복구한다.
- process kill 또는 OOM처럼 finally가 실행되지 않는 종료에서는 third_party config를 확인해야 한다.
- agent/tool/redundant/invalid action metric은 아직 baseline별 상세 trajectory에서 추출하지 않는다.
- Puppeteer는 매우 느릴 수 있으므로 GPT-Swarm, MASRouter, Puppeteer 순서로 붙이는 것이 안전하다.
