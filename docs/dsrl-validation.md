# Frozen SmolVLA + DSRL-SAC: 학습, checkpoint 선정, 공통 SAC 설정 문제

- 작성: 2026-11-28, 서버 `summit2`, 저장소 `~/steer-frozen-vla`, 브랜치 `dsrl/noise-sac`
- 범위: DSRL-SAC 3 seed 학습, validation에 의한 checkpoint 선정, validation 결과 분석. **Test는 아직 돌리지 않았다**(§7).
- 원본 결과: `results/dsrl_train_v1_s{42,43,44}/`, `results/dsrl_validation/` (서버, git 미추적). 요약 사본: [`docs/evidence/dsrl_validation/`](evidence/dsrl_validation/)
- 표기: **측정**은 로그와 평가에서 직접 계산한 값이고, **추정**은 측정에 근거한 해석이다(인과 미확정).
- 기준 문서: [`dynamic_libero_v2.md`](dynamic_libero_v2.md) §7(공통 프로토콜), `eval/residual-robustness` 브랜치의 `docs/residual-robustness-evaluation.md` §1.2, §7.2(checkpoint 선정 규칙)

## 0. 결론 요약

1. **학습과 선정은 끝났다.** `train_v1`에서 seed 42/43/44를 20,000 control step씩 학습했고, Residual의 `validation_rule.json`을 그대로 적용해 seed별 checkpoint를 골랐다(§3).
2. **Validation에서 DSRL의 학습 효과는 확인되지 않았다 (측정).**
   - 선정 checkpoint는 40 에피소드 중 36, 34, 35개 성공이다. 같은 에피소드에서 Frozen Base는 31개, 학습하지 않은 Gaussian latent는 34개다.
   - Base 대비 +3~+5개는 paired sign test로 유의하지 않다(p ≥ 0.13). 학습 없는 Gaussian latent와는 0~2개 차이다.
   - seed 43은 SAC 업데이트가 0번인 checkpoint(2k)가 선정됐다. 학습한 17개 후보 중 이를 넘은 것이 없다.
3. **끝까지 학습하면 Base보다 나빠진다 (측정).** 20k checkpoint는 27, 21, 21개 성공이다. seed 43/44는 Base보다 유의하게 낮다(둘 다 p = 0.006).
4. **Critic이 실패를 성공보다 높게 평가한다 (측정).** 15k 이후 세 seed 모두에서, 시간 초과로 끝난 에피소드의 Q가 성공한 에피소드의 Q보다 높다(§5.3).
   - **추정:** 고정 entropy 계수의 보너스가 시간 초과 bootstrap을 통해 누적되어 성공 보상(1)을 넘은 것이 원인이다. 후반 성능 하락과 시점이 맞지만, 설정을 고쳐 다시 학습하는 확인 실험은 아직 하지 않았다.
5. **같은 Q 징후가 Residual 학습 로그에도 있다 (측정).** Residual의 Q1 평균은 20k에서 2.0~2.1로, 가능한 최대 return 1의 두 배다(§5.4). 공통 `rl/` 설정에서 나오는 것이므로 두 방법에 같이 해당한다. Residual 성능에 준 영향은 분석하지 않았다.
6. **합의가 필요한 것이 세 가지 있다**(§6): seed 43 후보 범위, 공통 SAC 설정 수정 여부, test 진행 순서.

## 1. 설정

| 항목 | 값 |
|---|---|
| Base | `lerobot/smolvla_libero` (frozen), LIBERO-Object task 0, `n_action_steps=10`, max 280 step |
| RL 행동 | noise latent 6차원(팔). Flow-matching 초기 noise의 앞 6개 차원에 넣고 chunk 전체에 같은 값을 반복한다. 범위 ±2.5. 나머지 noise 차원은 에피소드 seed의 Gaussian 그대로다. |
| 결정 주기 | chunk(10 control step)마다 한 번. 에피소드당 최대 28번. |
| 관측 | 15차원: 로봇 상태 8 + latent 0일 때 VLA가 내는 첫 행동 7 (`proprio_ref`). 결정마다 VLA를 2번 호출한다. |
| SAC | 공통 `rl/` SAC(수정 없음), entropy coefficient 0.01 고정, 할인율 0.99(**결정 단위**), 나머지 SACConfig 기본값 |
| 학습 교란 | `train_v1`: 에피소드마다 ±x/±y 중 한 방향을 무작위로 골라 2cm 이동, Early 18cm 규칙 |
| 학습량 | 20,000 control step(약 2,030 결정), warmup 2,560 control step(Gaussian latent, 업데이트 없음), batch 256, 결정당 업데이트 10번, sparse success reward |
| Seed | 42, 43, 44 |
| 실행 | `train_dsrl.py --preset train_v1 --seed <seed> --run-dir results/dsrl_train_v1_s<seed>` |

- Residual과 다른 점:
  - 할인율 0.99가 결정 단위로 적용된다(Residual은 control step 단위).
  - warmup이 2,560 control step이다(Residual은 1,000). batch 256을 채우려면 256 결정이 필요하기 때문이다.
  - 그래서 2k checkpoint는 업데이트 0번 상태다(§3.3).
- 세 run의 `base_config.json`은 seed와 run 경로만 다르다.

## 2. 학습 (측정, `T_training_summary.csv`)

| Seed | Control step | 결정 | SAC 업데이트 | VLA 호출 | 학습 성공 에피소드 | 최종 entropy | 최종 Q1 | 후반 critic loss |
|---|---|---|---|---|---|---|---|---|
| 42 | 20,002 | 2,026 | 17,710 | 4,078 | 80/105 | 2.07 | 1.27 | 0.0017 |
| 43 | 20,005 | 2,026 | 17,710 | 4,082 | 74/103 | 1.07 | 1.31 | 0.0016 |
| 44 | 20,007 | 2,033 | 17,780 | 4,095 | 79/107 | 0.94 | 1.36 | 0.0018 |

- 세 run 모두 오류 없이 끝났고, VLA는 frozen 상태를 유지했다(`summary.json`의 `vla_frozen: true`).
- 학습 성공률은 탐색 noise가 섞인 값이라 성능 지표로 쓰지 않는다.
- **Q1 평균이 1을 넘는다.** Sparse reward의 최대 return은 1이다. 구간별 평균은 0.46~0.52(업데이트 초반) → 1.07~1.09 → 1.23~1.26 → 1.27~1.30이다. §5에서 다룬다.

## 3. Validation과 checkpoint 선정

### 3.1 적용한 규칙

Residual의 `configs/validation_rule.json`을 바꾸지 않고 적용했다.

- 후보: 2k~20k control step, 1k 간격, 19개. DSRL에서는 `ckpt_00000200.pt` ~ `ckpt_00002000.pt`다(파일 이름의 숫자는 결정 수).
- 평가: ±x/±y 2cm × 각 10 에피소드, eval seed 1000(에피소드 i의 seed는 1000+i, init state i), 결정론적 행동.
- 선택: 40 에피소드 중 성공 수가 가장 높은 checkpoint. 동점이면 나중 checkpoint.
- 57개 평가 모두 오류와 재시도 없이 끝났다. 교란은 2,280 에피소드 전부에서 적용됐다.

### 3.2 점수 곡선 (측정, 40 에피소드 중 성공 수, 2k→20k)

- seed 42: 33 35 28 23 32 31 33 32 32 32 33 36 34 **36** 31 31 27 30 27
- seed 43: **34** 33 26 31 32 27 31 31 29 31 29 33 30 30 29 29 28 32 21
- seed 44: 28 14 30 34 27 **35** 32 33 32 29 32 31 32 31 30 27 24 24 21

### 3.3 선정 결과 (측정, `runs/s*/validation/selection.json`)

| Seed | 선정 checkpoint | Control step | SAC 업데이트 | 점수 | 20k 점수 | 19개 평균 |
|---|---|---|---|---|---|---|
| 42 | `ckpt_00001500` | 14,780 | 12,450 | 36 | 27 | 31.4 |
| 43 | `ckpt_00000200` | 1,972 | **0** | 34 | 21 | 29.8 |
| 44 | `ckpt_00000700` | 6,888 | 4,450 | 35 | 21 | 28.7 |

- seed 42는 13k와 15k가 36점 동점이라 규칙대로 15k를 골랐다.
- **seed 43의 선정 checkpoint는 학습되지 않은 actor다.**
  - 업데이트는 256번째 결정(약 2,560 control step)부터 시작한다. 2k checkpoint의 actor는 초기화 상태 그대로이고, 결정론적 출력은 0에 가까운 latent다(평균 절댓값 0.14~0.19).
  - Residual 규칙은 같은 이유로 1k checkpoint를 후보에서 뺐다("end of zero-residual warmup, no SAC update yet").
  - 업데이트 0번인 후보를 빼면 seed 43은 `ckpt_00001300`(약 13k, 33점)이 된다. seed 42/44의 선정은 바뀌지 않는다.
  - `selection.json`에는 규칙을 글자 그대로 적용한 결과(`ckpt_00000200`)를 기록했다. 어느 쪽을 공식으로 할지는 §6에서 정한다.

## 4. Validation 분석

### 4.1 기준 정책 (측정, `V_reference.csv`)

선정된 checkpoint의 점수를 해석하려고, 같은 40 에피소드에서 두 정책을 평가했다.

| 정책 | +x 2cm | −x 2cm | +y 2cm | −y 2cm | 합계 |
|---|---|---|---|---|---|
| Frozen Base | 8 | 8 | 7 | 8 | 31 |
| Gaussian latent (학습 없음, 결정마다 무작위 latent) | 7 | 10 | 8 | 9 | 34 |

- 두 평가 모두 교란이 40/40 적용됐고, checkpoint 평가와 초기 상태 hash가 같다.

### 4.2 Base와의 paired 비교 (측정, `V_paired_vs_base.csv`)

| 정책 | 점수 | Base만 실패 | Base만 성공 | Sign test p |
|---|---|---|---|---|
| Gaussian latent | 34 | 4 | 1 | 0.38 |
| seed 42 선정 (15k) | 36 | 6 | 1 | 0.13 |
| seed 43 선정 (2k, 업데이트 0) | 34 | 4 | 1 | 0.38 |
| seed 43 대안 (13k) | 33 | 3 | 1 | 0.63 |
| seed 44 선정 (7k) | 35 | 6 | 2 | 0.29 |
| seed 42 최종 (20k) | 27 | 2 | 6 | 0.29 |
| seed 43 최종 (20k) | 21 | 1 | 11 | 0.006 |
| seed 44 최종 (20k) | 21 | 1 | 11 | 0.006 |

- 선정 checkpoint가 Base보다 높은 것은 유의하지 않다. 19개 후보 중 최고점을 고른 값이라 실제 성능보다 높게 나오는 편향도 있다.
- 후보 57개 전체로 보면 Base보다 높은 것이 23개, 같은 것이 9개, 낮은 것이 25개다.
- 최종 checkpoint가 Base보다 낮은 것은 seed 43/44에서 유의하다. 보정은 하지 않았다(탐색 분석).

### 4.3 학습 구간별, 방향별 (측정, `V_phase_means.csv`)

| 구간 | Checkpoint 수 | 평균 점수 | +x | −x | +y | −y | Latent 치우침 |
|---|---|---|---|---|---|---|---|
| 2k (업데이트 0) | 3 | 31.7 | 8.7 | 8.7 | 5.7 | 8.7 | 0.49 |
| 3k~5k | 9 | 28.2 | 7.3 | 7.6 | 6.7 | 6.7 | 1.48 |
| 6k~10k | 15 | 31.3 | 7.0 | 9.3 | 6.7 | 8.3 | 0.70 |
| 11k~15k | 15 | 31.9 | 7.4 | 9.5 | 6.5 | 8.5 | 0.69 |
| 16k~20k | 15 | 27.4 | 5.5 | 8.9 | 5.5 | 7.5 | 0.91 |

- 방향별 값은 10 에피소드 중 평균 성공 수다. "Latent 치우침"은 결정론적 latent의 차원별 평균이 이루는 벡터의 노름이다(§4.6).
- 업데이트 시작 직후(3k~5k)에 한 번 떨어지고, 6k~15k에는 Base 수준으로 돌아오고, 16k 이후 다시 떨어진다.
- 후반 하락은 +x와 +y에서 난다. −x는 끝까지 유지된다.

### 4.4 에피소드별 (측정, `V_episode_outcomes.csv`)

- 40개 중 4개는 거의 모든 정책이 실패한다: +x의 에피소드 2, 9와 +y의 에피소드 2, 8. Checkpoint 57개 중 성공 비율이 7%, 26%, 2%, 5%다. Base와 Gaussian latent도 넷 다 실패한다.
  - 따라서 이 validation set의 현실적 상한은 36~37점이다. seed 42의 36점은 상한에 가깝다.
- Base가 실패한 나머지 5개는 checkpoint의 39~93%가 성공하고, 그중 4개는 Gaussian latent도 성공한다.
  - **추정:** Base 대비 이득은 학습한 대응이라기보다 noise를 다르게 넣은 것 자체에서 나온다.

### 4.5 실패 유형 (측정)

- 모든 정책에서 실패는 전부 280 step 시간 초과다. 조기 종료 실패는 없다.
- 성공 에피소드의 평균 길이는 Base 152 step, 선정 checkpoint 152~158 step, 최종 checkpoint 150~154 step이다. DSRL이 더 빨리 성공하지는 않는다.

### 4.6 정책이 고른 latent (측정, `V_scores_by_checkpoint.csv`)

- 범위 한계(±2.5) 근처(절댓값 2.4 초과)의 latent는 57개 checkpoint 모두에서 0%다.
- 학습한 checkpoint 54개에서 latent 치우침과 점수의 상관은 −0.50이다.
  - 치우침 0.9 미만: 34개, 평균 31.3점
  - 0.9~1.3: 13개, 평균 28.6점
  - 1.3 이상: 7개, 평균 25.3점
- 20k checkpoint의 치우침은 1.28, 1.39, 1.15로, 13k(0.77, 0.85, 0.62)보다 크다.
- **추정:** 정책이 noise를 한쪽으로 일관되게 밀수록 성공률이 떨어진다. 상태에 따라 latent를 바꾸는 것이 아니라 고정된 편향을 더하는 쪽으로 학습이 진행된 것으로 보인다.

## 5. 공통 SAC 설정 문제: entropy 보너스와 시간 초과 bootstrap

### 5.1 구조 (측정, 코드)

- Critic target은 soft value를 쓴다(`rl/sac.py`).
  ```python
  target_v = torch.min(tq1, tq2) - alpha * next_logp
  target_q = rew + cfg.gamma * (1.0 - done) * target_v
  ```
- `done`은 성공 종료일 때만 1이다. 시간 초과는 0으로 저장되어 bootstrap한다(`rl/replay_buffer.py`: "`dones` = terminated only (NOT truncated)").
- Entropy coefficient는 0.01로 고정되어 있고, 관측에는 남은 시간이 들어 있지 않다.
- 따라서 매 전이에 `alpha × entropy`의 보너스가 붙고, 성공하면 그 흐름이 끊기며, 성공하지 않으면 계속 이어지는 것으로 계산된다.

### 5.2 크기 (측정한 entropy로 계산)

성공하지 않고 계속 가는 경우의 soft value 고정점은 `gamma × alpha × entropy / (1 − gamma)`이다.

| | Entropy | 전이당 보너스 | 누적 고정점 | 성공의 가치 |
|---|---|---|---|---|
| DSRL (결정 단위 할인) | 1.7~3.6 (10k~20k checkpoint) | 0.017~0.036 | 1.6~3.5 | 15 결정 뒤 성공: 약 0.87 + 보너스 0.36 ≈ 1.2 |
| Residual (control step 단위 할인) | 3.9~4.1 | 약 0.04 | 3.8~4.0 | 150 step 뒤 성공: 보상 0.99^150 ≈ 0.22 + 보너스 약 3.1 ≈ 3.3 |

- 두 방법 모두 누적 고정점이 성공의 가치보다 크다. 이 목적함수에서는 성공하지 않는 쪽이 더 값지다.
- 일반적으로 k 전이 뒤에 성공할 때의 이점은 `gamma^k × (1 − 누적 고정점)`이다. 누적 고정점이 1을 넘으면 부호가 뒤집힌다.
- DSRL에서 성공 에피소드 전이의 Q가 1.21~1.24에서 멈춘 것(§5.3)은 위 계산값 1.2와 맞는다.

### 5.3 DSRL critic 직접 확인 (측정, `Q_critic_by_outcome.csv`)

저장된 checkpoint의 critic으로 replay buffer의 전이를 평가하고, 그 전이가 속한 에피소드의 결과로 나눴다. 값은 `min(Q1, Q2)`의 평균이다.

| Checkpoint | Seed | 성공 에피소드 전이 | 실패 에피소드 전이 | 성공 마지막 전이 (target 1.0) | 시간 초과 전이 |
|---|---|---|---|---|---|
| 5k | 42 / 43 / 44 | 0.88 / 0.86 / 0.84 | 0.75 / 0.77 / 0.75 | 1.01 / 1.00 / 0.99 | 0.69 / 0.80 / 0.75 |
| 10k | 42 / 43 / 44 | 1.20 / 1.21 / 1.17 | 1.22 / 1.25 / 1.20 | 1.00 / 1.00 / 1.00 | 1.17 / 1.23 / 1.15 |
| 15k | 42 / 43 / 44 | 1.22 / 1.23 / 1.22 | 1.36 / 1.33 / 1.33 | 1.00 / 1.00 / 1.00 | 1.40 / 1.34 / 1.37 |
| 20k | 42 / 43 / 44 | 1.21 / 1.24 / 1.22 | 1.43 / 1.40 / 1.48 | 1.00 / 1.02 / 1.00 | 1.51 / 1.47 / 1.66 |

- 5k에서는 성공 에피소드의 Q가 더 높다(정상).
- 10k에서 두 값이 같아지고, 15k부터 세 seed 모두 실패 에피소드의 Q가 더 높다.
- 20k에서 시간 초과 전이의 Q는 1.47~1.66으로, 성공 마지막 전이(1.0)보다 0.5 이상 높다.
- 10k 이후 buffer 상태의 98% 이상에서 soft value가 1을 넘는다.

### 5.4 Residual 학습 로그 (측정, `Q_residual_reference.csv`)

서버의 Residual run(`residual_6d_rand_fixa001_2cm_s{42,43,44}_20k/train.csv`)에서 Q1 평균을 구간별로 계산했다.

| Env step | Q1 평균 (seed 42 / 43 / 44) | Entropy |
|---|---|---|
| 1k~5k | 0.32 / 0.46 / 0.35 | 3.6~3.8 |
| 5k~10k | 1.03 / 1.13 / 1.03 | 4.0~4.1 |
| 10k~15k | 1.63 / 1.69 / 1.63 | 3.9~4.0 |
| 15k~20k | 2.02 / 2.06 / 2.01 | 3.9~4.0 |

- Residual 문서 §2.1의 "최종 Q1 2.14~2.19"와 같은 값이다.
- Q1 평균이 최대 return 1을 5k~10k에 넘고, 20k까지 계속 오른다. 누적 고정점(약 4)을 향해 가는 모습과 맞는다.
- Residual critic을 에피소드 결과별로 나눠 보는 분석(§5.3과 같은 것)은 하지 않았다. Residual 성능에 준 영향도 분석하지 않았다.
  - 참고로 Residual 문서는 validation 점수가 6k~7k에서 최고였고 20k checkpoint가 2cm에서 Base 이하였다고 적고 있다(§6의 1번).

### 5.5 해석과 확인 방법

- **측정:** 코드 구조(§5.1), Q가 1을 넘는 것(§2, §5.4), DSRL critic이 15k부터 실패를 성공보다 높게 평가하는 것(§5.3), DSRL validation이 16k 이후 떨어지는 것(§4.3).
- **추정:** DSRL의 후반 하락은 이 보너스 누적 때문이다. Actor는 Q를 올리는 방향으로 갱신되므로, critic이 실패 쪽을 높게 평가하면 성공에서 멀어지는 latent 편향이 생길 수 있다(§4.6).
- **확인되지 않은 것:** 인과. 설정을 고쳐 다시 학습했을 때 후반 하락이 사라지는지 봐야 한다.
- 수정 후보(모두 공통 `rl/` 또는 공통 설정을 바꾼다):
  1. **Critic target에서 entropy 항을 뺀다.** Q가 순수한 task return(≤ 1)이 된다. Entropy 항은 actor loss에만 남긴다. 구조 문제를 직접 없애는 방법이라 이것을 권한다.
  2. 시간 초과를 종료로 처리한다. DSRL에서는 실패의 보너스 합이 약 0.7로 유한해져 성공(약 1.2)이 다시 우위가 된다. Residual에서는 280 step 동안의 보너스 합(약 3.7)이 150 step 뒤 성공의 가치(약 3.3)보다 여전히 커서 해결되지 않는다.
  3. Entropy coefficient를 낮춰 누적 고정점을 1 아래로 내린다. 순서는 바로잡히지만 성공의 이점이 `1 − 누적 고정점` 비율로 줄어든다. 0.001이면 Residual의 누적 고정점은 약 0.4이고 이점은 60%로 줄어든다. Entropy가 달라지면 다시 1을 넘을 수 있다.
- DSRL 한 seed의 재학습은 약 30분이다.

## 6. 합의가 필요한 사항

1. **seed 43의 공식 checkpoint.** 규칙 그대로면 `ckpt_00000200`(업데이트 0번), 업데이트 0번인 후보를 빼면 `ckpt_00001300`이다. Test 숫자를 보기 전에 정해야 한다. 두 checkpoint 모두 test 대기열에 넣어 두었다.
2. **공통 SAC 설정을 고칠지.** 고친다면 두 방법을 같은 설정으로 다시 학습해야 비교가 공정하다. 고치지 않는다면 이 문제를 두 방법의 공통 한계로 문서에 적는다.
3. **Test 진행 순서.** 지금 checkpoint로 test를 먼저 끝낼지, §5의 확인 실험을 먼저 할지.

## 7. Test 계획 (대기 중)

- Test 에피소드를 하나도 돌리기 전에 계획을 기록했다: [`queue/test_plan.json`](evidence/dsrl_validation/queue/test_plan.json) (2026-11-28 22:34 작성).
- 조건: `primary`(0cm, ±x/±y 2cm)와 `distance_4cm`(±x/±y 4cm), 각 50 에피소드, eval seed 0, 결정론적 행동. Residual 문서 §7.2의 9개 조건과 같다.
- 정책 9개:
  - `base`, `gaussian`
  - `s42_val`(15k), `s43_val_rule`(2k), `s43_val_trained`(13k), `s44_val`(7k)
  - `s42_final`, `s43_final`, `s44_final`: 마지막 validation 후보인 `ckpt_00002000`(약 19.7k control step). 보조 보고용이다.
- 81개 작업(정책 × 조건)이 GPU가 비면 시작하도록 서버에서 대기 중이다. 이 문서 작성 시점에는 시작되지 않았다.
- 교란 후 개입 제거(`--post-perturb base`)와 timing, chunk phase 보조 세트는 아직 대기열에 넣지 않았다.

## 8. 한계

- Validation은 조건당 10 에피소드, 합계 40 에피소드다. 3~4개 차이는 우연으로도 생긴다.
- Validation(eval seed 1000)과 test(eval seed 0)는 noise seed가 다르지만 init state 0~9가 겹친다(Residual과 같은 조건).
- §4.2의 sign test는 다중 비교 보정을 하지 않은 탐색 분석이다.
- §5.3은 학습 중 모은 buffer 전이에 대한 critic 값이다. Validation 궤적에서의 critic 값은 저장하지 않았다.
- §5의 인과는 확인 실험 전이다. Residual에 대해서는 Q1 평균만 봤다.
- 하이퍼파라미터 탐색은 하지 않았다. latent 6차원, 범위 ±2.5, 결정당 업데이트 10번, 관측 15차원의 설정 하나에 대한 결과다.
- 평가 중 root 사용자의 vLLM 서버가 GPU를 점유해, validation과 기준 평가를 GPU가 빌 때 자동으로 시작하는 방식으로 돌렸다. 결과에는 영향이 없다(완료된 보고서만 집계).

## 9. 파일 및 재현

### 9.1 코드 (이 브랜치)

| 파일 | 역할 |
|---|---|
| `models/noise_wrapper.py`, `policies/noise_policy.py` | Latent를 받는 SmolVLA wrapper와 chunk 단위 env |
| `train_dsrl.py` | 공통 `rl/` SAC로 DSRL 학습 |
| `scripts/eval_dsrl.py` | Base, Gaussian latent, checkpoint 평가 |
| `scripts/analyze_dsrl_validation.py` | 이 문서의 표(`V_*`, `T_*`, `Q_*`)를 만든다. CPU에서 돈다. |

- `rl/`, `envs/`, Residual 코드는 수정하지 않았다.

### 9.2 증거 (`docs/evidence/dsrl_validation/`, 서버 원본의 복사본)

- `V_scores_by_checkpoint.csv`: checkpoint 57개의 방향별 점수, Base와의 paired 차이, latent 통계.
- `V_phase_means.csv`, `V_reference.csv`, `V_paired_vs_base.csv`, `V_episode_outcomes.csv`: §4의 표.
- `T_training_summary.csv`: §2의 표.
- `Q_critic_by_outcome.csv`, `Q_residual_reference.csv`: §5의 표. 앞의 파일은 3k~20k checkpoint 전부(54행)를 담고 있다.
- `runs/s{42,43,44}/`: `base_config.json`, `config.json`, `summary.json`, `episodes.csv`, `train.csv`, `validation/selection.json`, `validation/validation_scores.csv`.
- `reference/`: Base와 Gaussian latent의 validation 보고서(에피소드별 기록 포함).
- `queue/`: validation, 기준 평가, test를 돌린 실행 스크립트와 test 계획(`run_*.sh`, `jobs.txt`, `test_plan.json`).
- Checkpoint 평가 원본(57개 JSON)과 checkpoint 파일은 서버 `results/dsrl_train_v1_s*/`에 있다. 크기 때문에 git에는 넣지 않았다.

### 9.3 재현 명령 (서버, 저장소 루트)

```bash
source scripts/server_env.sh <gpu>

# 학습 (seed당 약 30분)
"$PY" -B train_dsrl.py --preset train_v1 --seed 42 --run-dir results/dsrl_train_v1_s42

# Validation: checkpoint 하나
"$PY" -B scripts/eval_dsrl.py --policy results/dsrl_train_v1_s42/checkpoints/ckpt_00001500.pt \
  --set primary --condition +x2 --condition=-x2 --condition +y2 --condition=-y2 \
  --episodes 10 --eval-seed 1000 --out results/dsrl_train_v1_s42/validation/ckpt_00001500.json

# 기준 정책 (같은 40 에피소드): --policy base 또는 --policy gaussian

# 분석 (CPU)
"$PY" -B scripts/analyze_dsrl_validation.py --residual-results ~/rf/steer-frozen-vla/results
```

- 19개 checkpoint를 차례로 도는 실행 스크립트는 `queue/run_validation.sh`다. 저장소 경로가 `/home/sylee/steer-frozen-vla`로 고정되어 있다.
