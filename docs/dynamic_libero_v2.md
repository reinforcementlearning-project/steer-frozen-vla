# Dynamic LIBERO v2: Residual-SAC·DSRL-SAC 공통 교란 환경

- 작성: 2026-11-24, 브랜치 `env/dynamic-libero-v2` (main `0d7c237` 기준)
- 코드:
  - [`envs/dynamic_libero.py`](../envs/dynamic_libero.py): wrapper
  - [`envs/dynamic_libero_config.py`](../envs/dynamic_libero_config.py): 설정, 프리셋, 환경 생성 함수
  - [`configs/dynamic_libero_v2.yaml`](../configs/dynamic_libero_v2.yaml): 학습·평가 조건
- 검증 결과 원본(서버): `results/dynamic_libero_v2/validation/`. 요약 사본: [`docs/evidence/dynamic_libero_v2/`](evidence/dynamic_libero_v2/)
- 기존 `envs/perturbation.py`(v1 `DynamicLibero`, `PerturbationConfig`)는 **수정하지 않았다.** v2는 별도 모듈이고, 기본 설정에서 v1과 같은 시뮬레이터 호출과 결과를 낸다(§6).

## 0. 요약: 팀 합의 사항

| 항목 | 결정 | 구현 |
|---|---|---|
| 초기 물체 위치 | LIBERO 원본 init state가 기본이다. 위치 무작위화는 OOD 평가 옵션으로만 쓴다. | `spatial.mode: none`이 기본값이다. `region_uniform`은 `eval_ood` split(`ood_spatial`)에서만 쓴다. |
| 교란 시점 | Early, Middle, Late를 구현한다. 정책마다 실제 발생 step이 다르므로 기록한다. | `perturbation.timing`. 발생 step, EEF 거리, 적용 여부, 거부 사유, 미발생 사유를 `info`에 기록한다(§3). |
| Episode 제한 | 공식 학습·평가는 280 step이다. 400 step은 진단 전용이다. | 모든 학습·공식 평가 설정은 `max_steps: 280`이다. `horizon400`은 `diagnostic` split이다. |
| 하위 호환 | 기존 18cm 교란의 기본 동작과 재현성을 유지한다. 신규 기능은 opt-in이다. | 기본값 = v1 규칙. 시뮬레이터와 Frozen VLA에서 에피소드 단위로 동일함을 확인했다(§6). |
| 공통 인터페이스 | 설정 항목과 `info` 키를 먼저 확정한다. | §2, §3에 정리했다. 키는 `INFO_KEYS`, `STATUS_KEYS`, `EVENT_KEYS` 상수로 고정하고 테스트로 검증한다. |
| 학습·평가 분리 | 학습 조건: Early 18cm, ±x/±y 무작위, 2cm, 280 step. Middle, Late, 4cm는 학습하지 않은 보조 평가다. | 학습은 `train_v1`, 공식 평가는 `primary`, 보조 평가는 `eval_secondary`, OOD는 `eval_ood`, 진단은 `diagnostic`이다. |

## 1. 기존 환경(v1)의 한계와 v2의 대응

| v1 한계 (측정) | v2 대응 |
|---|---|
| Task 0의 목표 물체가 50개 init state 모두에서 같은 위치에 있다(BDDL 영역 정중앙, 변동 0mm). | 원본은 그대로 유지한다. OOD 평가용으로 seed 기반 위치 이동을 opt-in으로 추가했다. |
| 교란이 사실상 EEF–물체 거리 18cm 진입 순간에만 발생한다(실측 17.1~18.0cm, step 27~39). | Early, Middle, Late 거리 구간을 추가했다. chunk phase 지정도 할 수 있고 기록된다. |
| ±y 4cm는 280 step 제한에 의해 결과가 잘린다. | 280 step을 공식 기준으로 유지하고, 400 step은 `diagnostic` 평가 세트로 분리했다. |
| 학습·평가 조건과 실행 방식이 코드 곳곳에 흩어져 있다. | YAML 한 곳에 split별로 모으고, 환경 생성 함수 `make_dynamic_libero(cfg)` 하나로 통일했다. |

## 2. 설정 항목 (확정)

`DynamicLiberoConfig`의 각 항목이다. YAML의 `defaults` 아래에 같은 이름으로 쓴다.

| 키 | 기본값 | 의미 |
|---|---|---|
| `name`, `split` | `custom`, `legacy` | split은 `train` / `eval_primary` / `eval_secondary` / `eval_ood` / `diagnostic` / `legacy` 중 하나 |
| `suite`, `task_id`, `target` | `libero_object`, 0, `alphabet_soup_1` | Task와 교란 대상 |
| `max_steps` | 280 | Episode 제한. 400은 `diagnostic`에서만 쓴다. |
| `n_action_steps` | 10 | VLA chunk 실행 길이. chunk phase 계산에 쓴다. |
| `episodes`, `eval_seed` | 평가 세트에서 50, 0 | 에피소드 i의 reset·noise seed는 `eval_seed + i`이고, LIBERO init state는 i번이다. |
| `perturbation` | 아래 표 | `null`이면 교란이 없다. |
| `spatial` | `mode: none` | 초기 위치 이동. OOD 전용이다. |

**`perturbation` (`PerturbationSpec`, v2에서 `PerturbationConfig` 역할)**

| 키 | 기본값 | 의미 |
|---|---|---|
| `timing` | `legacy` | `legacy` [0.05, 0.18]m, `early` [0.15, 0.18], `middle` [0.10, 0.15], `late` [0.05, 0.10]. 조건을 만족하는 첫 step에 교란한다. |
| `window_m` | `null` | 사용자 지정 [min, max]. 지정하면 `timing`보다 우선한다. |
| `directions` | `["+y"]` | `+x`, `-x`, `+y`, `-y`. 2개 이상이면 에피소드마다 `rng([seed, 0x5EED])`로 하나를 뽑는다. |
| `distances_m` | `[0.02]` | 0~0.10m. 2개 이상이면 에피소드마다 `rng([seed, 0xD157])`로 하나를 뽑는다. |
| `chunk_phase` | `null` | `pre_replan`(0), `post_replan`(1), `mid_chunk`(n/2), 또는 정수. 지정하면 해당 phase의 step에서만 교란한다. |
| `earliest_step` | 10 | 이 step 이전에는 교란하지 않는다(v1과 같음). |
| `lift_tolerance_m` | 0.015 | 물체 높이가 이만큼 변하면 "들림"으로 보고 교란하지 않는다. |
| `workspace_limit_m` | 0.45 | 이동 후 \|x\|, \|y\|가 이 값을 넘으면 거부한다. |
| `penetration_tolerance_m` | 0.003 | 이동 후 이보다 깊은 관통이 생기면 원위치로 되돌리고 거부한다. |

- 교란 조건은 다음을 모두 만족하는 첫 step이다. 에피소드당 최대 1회 시도한다.
  - step ≥ `earliest_step`
  - 실제 EEF–물체 거리가 구간 안
  - 로봇과 접촉하지 않음
  - 물체가 들리지 않음
  - (지정한 경우) chunk phase 일치
- `0cm`는 v1과 같다. 위치는 그대로 두고 속도를 유지한다.

**`spatial` (`SpatialSpec`, OOD 전용)**

| 키 | 기본값 | 의미 |
|---|---|---|
| `mode` | `none` | `region_uniform`이면 원본 init state를 로드하고 settle한 뒤, 목표 물체를 xy 방향으로 이동한다. |
| `max_offset_m`, `min_offset_m` | 0.025, 0 | offset은 축마다 U(−max, max)에서 뽑는다(`rng([seed, 0x51A7])`). |
| `region_margin_m` | 0 | BDDL 초기 영역 경계로부터의 여유 |
| `max_attempts`, `settle_steps`, `settle_tolerance_m` | 50, 10, 0.005 | 유효성 검사(§4)를 통과할 때까지 다시 뽑는다. 끝내 실패하면 오류를 낸다. |

## 3. `info` 반환 키 (확정)

모든 `step()`의 `info`에 아래 키가 붙는다. `reset()`에는 `dynamic_libero`만 붙는다.

| 키 | 내용 |
|---|---|
| `perturb_events` | 이벤트 목록(0개 또는 1개). v1과 같은 키다. |
| `perturb_step` | 교란 시도 step. 없으면 −1이다. |
| `perturb_dist`, `perturb_axis`, `perturb_sign` | 이번 에피소드에 뽑힌 크기(m), 축(0 = x, 1 = y), 부호(±1) |
| `dynamic_libero` | 에피소드 상태. **에피소드 마지막 step의 값이 최종값이다.** |

**`info["dynamic_libero"]`**
- 공통 키: `config`, `split`, `episode_seed`, `choice`(`direction`, `axis`, `sign`, `distance_m`), `spatial`, `status`
- `status` 값:
  - `applied`: 이동 완료
  - `rejected`: 시도했으나 거부됨
  - `not_triggered`: 시도 조건이 한 번도 성립하지 않음
  - `disabled`: 교란 없음
- `applied`와 `rejected`일 때 추가 키: `reason`, `step`, `chunk_phase`, `eef_distance`, `diagnostics`
- `not_triggered`일 때 추가 키: `reason`, `diagnostics`
- 거부 사유: `conservative_workspace_boundary`, `new_penetration_or_robot_contact`
- 미발생 사유:
  - `window_not_reached`: 구간에 도달하지 않음
  - `window_skipped_between_steps`: 한 step 사이에 구간을 건너뜀
  - `blocked_robot_contact`, `blocked_object_lifted`, `blocked_chunk_phase_mismatch`: 구간 안에서 해당 사유로 막힘
  - `not_checked_yet_or_episode_ended_before_earliest_step`: 아직 검사 전이거나 `earliest_step` 전에 에피소드가 끝남
- `diagnostics`:
  - `steps_checked`, `steps_in_window`, `first_window_step`, `min_distance_m`
  - `blocked`: 사유별 횟수

**`info["perturb_events"][0]` 키**

| 키 | 내용 |
|---|---|
| `step` | 교란 시도 step |
| `eef_distance` | 시도 시점의 EEF–물체 거리 |
| `applied`, `rejection` | 적용 여부와 거부 사유 |
| `before`, `after` | 이동 전후 위치 |
| `direction`, `distance_m` | 방향과 크기 |
| `timing`, `window_m` | 시간 구간 이름과 범위 |
| `chunk_phase` | step mod n |
| `steps_until_replan_obs` | VLA가 이동된 장면을 볼 때까지 남은 step 수 |
| `chunk_phase_target` | 지정한 phase (없으면 null) |
| `last_reported_replan_step`, `chunk_phase_from_reported_replan` | 선택적 hook으로 보고받은 재계획 시점과, 그로부터 계산한 phase |
| `robot_contact_before` | 시도 직전 로봇 접촉 여부 |
| `config` | 이번 에피소드의 교란 설정 |
| `actual_displacement_m`, `actual_displacement_vector_m` | 적용된 경우에만: 실제 이동 거리와 벡터 |

- 이 키들은 `envs/dynamic_libero.py`의 `INFO_KEYS`, `STATUS_KEYS`, `EVENT_KEYS`, `APPLIED_EVENT_KEYS`, `NOT_TRIGGERED_REASONS`, `REJECTION_REASONS` 상수로 고정되어 있다.
- `tests/test_dynamic_libero_v2.py::test_info_contract_keys`가 적용, 거부, 미발생, 비활성 네 경우를 모두 검사한다.
- 시뮬레이터 좌표는 교란 일정을 정하고 유효성을 검사하는 데만 쓴다. **정책 관측에는 넣지 않는다.**

## 4. 설계 세부

### 4.1 Action chunk phase

- 측정 근거: 기존 평가 90개 파일의 VLA 호출 92,418회가 모두 env_step이 10의 배수인 시점에 일어났다.
  - `FrozenSmolVLA`는 큐 방식이다. step s(1부터 셈)가 끝난 뒤의 관측에서, `s % 10 == 0`일 때만 재계획한다.
  - 따라서 v2 wrapper는 `chunk_phase = step % n_action_steps`로 계산한다.
  - 이번 검증 rollout에서도 550/550 에피소드에서 재계획 주기가 맞았다(§5.3).
- phase별 의미:

  | phase | 이름 | VLA가 이동된 장면을 보기까지 |
  |---|---|---|
  | 0 | `pre_replan` (다음 재계획 직전) | 바로 다음 관측에서 재계획하므로 즉시 |
  | 1 | `post_replan` (재계획 직후) | 새 chunk가 시작된 직후라 9 step 동안 보지 못함 |
  | 5 | `mid_chunk` (chunk 중간) | 5 step 동안 보지 못함 |

- **불가능한 부분:** 환경은 정책 wrapper의 재계획 시점을 강제할 수 없다.
  - 모듈러 계산은 "재계획이 step 0부터 n step마다 일어난다"는 가정에 의존한다.
  - DSRL wrapper가 다른 주기로 재계획하면 phase 지정의 의미가 달라진다.
  - 이를 확인하려고 선택적 hook `sim.notify_replan(env_step)`을 두었다. 정책 wrapper가 재계획할 때 호출하면, 실제 재계획 기준 phase(`chunk_phase_from_reported_replan`)가 이벤트에 함께 기록된다.

### 4.2 초기 위치 이동 (OOD 옵션)

- 순서:
  1. LIBERO 원본 init state를 그대로 로드하고 settle 10 step을 실행한다.
  2. 목표 물체를 xy 방향으로 이동한다.
  3. 아래 조건을 모두 통과하면 채택하고, 하나라도 실패하면 sim state를 복원한 뒤 다시 뽑는다.
- 채택 조건:
  - 이동 후 xy가 목표 물체의 BDDL 초기 영역 안
  - 관통 없음
  - 로봇 접촉 없음
  - dummy action 10 step 동안 물체가 0.5cm 이상 움직이지 않음
  - 목표의 `on` predicate 유지
  - 이동 전 True였던 BDDL 초기 predicate가 False로 바뀌지 않음
  - goal이 이미 달성된 상태가 아님
- **측정:** LIBERO가 배포한 init state 자체가 BDDL 초기 predicate를 모두 만족하지는 않는다.
  - Task 0에서 cream_cheese, butter, basket의 `on` predicate는 원본에서 이미 False다.
  - basket은 영역 밖으로 최대 약 0.5cm 나가 있다.
  - 그래서 "모든 predicate가 True여야 한다"가 아니라 "이동이 새로 깨는 predicate가 없어야 한다"를 기준으로 삼았다.
- Task 0의 목표 영역은 5cm × 5cm이고 목표는 그 정중앙에 있다. 따라서 `max_offset_m: 0.025`이면 영역 안에서만 움직인다.
- **Task의 의미는 바뀌지 않는다.** 같은 물체를 같은 바구니에 넣는 작업이고, 초기 조건(`on target region`)도 유지된다.
- 원본 init state 경로는 그대로다. spatial을 끈 reset의 sim state hash가 변하지 않음을 테스트로 확인했다.
- 기록: `offset_m`, `attempts`, 사유별 거부 수, 이동 전 위치, settle 후 위치, 영역, seed, 원본에서 이미 False였던 predicate 목록.

## 5. 실제 시뮬레이터 검증

모든 run은 학습 없이 **Frozen SmolVLA**(residual 0)로 돌렸다.
- 조건마다 50 에피소드, eval seed 0(에피소드 i = init state i)이다.
- 방향을 무작위로 뽑은 run(`rand2`)에서는 방향이 +x 16, −x 11, +y 11, −y 12회 나왔다.
- 실행: `results/dynamic_libero_v2/validation/run_all.sh`
- 요약: `scripts/summarize_dynamic_libero_v2.py` → `summary.json`, `summary_runs.csv`, `summary_feasibility.csv`
- 성공률은 환경 검증용 참고값일 뿐이다. 방향 구성이 섞여 있어 정책 비교 지표가 아니다.

### 5.1 교란 시점: 적용률, 발생 step, EEF 거리 (측정)

| Run | 구간 / phase | 적용 / 거부 / 미발생 | 발생 step (min–중앙값–max) | EEF 거리 (m) | chunk phase 분포 (0..9) | 이동량 = 설정 | 성공 |
|---|---|---|---|---|---|---|---|
| `timing_early_rand2` | Early [0.15, 0.18] | 50 / 0 / 0 | 27–34–39 | 0.171–0.180 | 1,3,2,11,11,5,4,5,5,3 | 50/50 | 45/50 |
| `timing_middle_rand2` | Middle [0.10, 0.15] | 50 / 0 / 0 | 32–37–42 | 0.140–0.150 | 3,5,5,1,3,2,7,14,7,3 | 50/50 | 46/50 |
| `timing_late_rand2` | Late [0.05, 0.10] | 50 / 0 / 0 | 37–42–46 | 0.089–0.100 | 4,5,12,9,4,3,5,3,3,2 | 50/50 | 44/50 |
| `chunk_pre_rand2` | legacy + `pre_replan` | 50 / 0 / 0 | 30–40–40 | 0.081–0.175 | 50 at phase 0 | 50/50 | 43/50 |
| `chunk_post_rand2` | legacy + `post_replan` | 50 / 0 / 0 | 31–41–41 | 0.087–0.178 | 50 at phase 1 | 50/50 | 45/50 |
| `chunk_mid_rand2` | legacy + `mid_chunk` | 50 / 0 / 0 | 35–35–45 | 0.083–0.179 | 50 at phase 5 | 50/50 | 45/50 |

- **측정:** Frozen Base 궤적에서 Early, Middle, Late 모두 50/50 에피소드에서 교란이 적용됐다.
  - 거부와 미발생은 0건이다.
  - 각 구간에서 교란은 구간에 처음 진입한 step에서 일어났다.
- **Frozen 궤적 (무교란 run `monitor_0cm`, step별 거리 기록):**
  - 50/50 에피소드에서 로봇이 물체에 접촉했다. 첫 접촉 step의 중앙값은 49, 그때 거리의 중앙값은 0.020m다.
  - 최소 거리의 중앙값은 0.013m다.
  - 따라서 Late 교란(중앙값 step 42, 약 9.5cm)은 첫 접촉보다 약 7 step 앞선다. 즉 **"파지 직전"**이라는 의도와 맞다.
- **최종 구간값:** 초기 설계안(Early 15~18, Middle 10~15, Late 5~10cm)을 그대로 확정한다. 세 구간 모두 Frozen 궤적에서 발생 가능했다.
  - 다만 학습한 정책은 접근 속도가 다를 수 있다. 그래서 평가마다 `status`, `step`, `eef_distance`를 함께 보고한다.
- **chunk phase:**
  - legacy 구간(5~18cm)과 함께 쓰면 50/50 에피소드에서 지정한 phase에서 교란이 일어났다.
  - 대신 phase를 맞추느라 발생이 최대 9 step 늦어지고, 그만큼 EEF 거리가 퍼진다(0.08~0.18m).
  - 좁은 구간과 함께 쓰면 발생 가능 비율이 30~70%로 떨어진다(오프라인 추정, `summary_feasibility.csv`).

    | 구간 | `pre_replan` | `post_replan` | `mid_chunk` |
    |---|---|---|---|
    | Early | 34% | 30% | 62% |
    | Middle | 66% | 66% | 40% |
    | Late | 38% | 48% | 70% |

  - 따라서 평가 세트의 chunk phase 조건은 legacy 구간과만 조합했다.

### 5.2 초기 위치 다양화 (OOD 옵션, 측정)

| Run | 위치 이동 적용 | offset 범위 (cm, x / y) | 평균 시도 수 | 거부 | 교란 적용 | 발생 step | 성공 |
|---|---|---|---|---|---|---|---|
| `spatial_0cm` | 50/50 | x −2.42~+2.17, y −2.40~+2.46 | 1.02 | 영역 밖 1건 | 50/50 | 23–28.5–32 | 46/50 |
| `spatial_rand2` | 50/50 | 같음 (seed가 같음) | 1.02 | 영역 밖 1건 | 50/50 | 23–28.5–32 | 37/50 |

- 50개 seed 모두에서 유효한 위치를 찾았다. 관통, 불안정, predicate 위반에 의한 거부는 0건이다.
- 물체 위치가 달라지자 18cm 진입 step이 앞당겨졌다(중앙값 34 → 28.5).
- **OOD 옵션으로 쓸 수 있다.** 공식 학습·평가에서는 끈다(`mode: none`이 기본값).

### 5.3 VLA 재계획 주기

- 이번 검증 rollout 550 에피소드 모두에서 VLA 호출 step이 0, 10, 20, …이었다(`replan_cadence_ok` 550/550).
- 따라서 `chunk_phase = step % 10`은 실제 재계획 주기와 일치한다.

### 5.4 참고: 0cm ≠ 무교란

- `no_perturbation` 프리셋(교란 비활성)의 Frozen Base 0cm 성공은 45/50이고, 기존 `0cm` 조건은 47/50이다.
- 0cm 조건은 18cm 진입 시 "0cm 이동"을 실행한다. 이때 위치와 속도는 그대로지만 `sim.forward()`와 관측 강제 갱신이 일어나 궤적이 달라질 수 있다.
- 공식 0cm 조건은 기존과 같은 `0cm`을 쓴다. 무교란 run은 궤적 기록용 진단이다.

## 6. 하위 호환성

- 기본 설정(`timing: legacy`, chunk phase 없음, spatial 없음)의 v2는 v1 `DynamicLibero`와 다음이 모두 같다.
  - 시뮬레이터 호출 순서: `_get_observations` 호출 시점과 횟수
  - 이동 pose
  - 방향 난수 stream: `rng([seed, 0x5EED])`. Residual `--random-direction` run과 같다.
- 확인 방법:
  - 단위 테스트(fake LIBERO): 4방향 × 3크기 × 2 seed에서 이벤트, pose, 호출 순서가 같다.
  - 시뮬레이터 테스트(scripted 접근): +y 2cm, −x 4cm, 0cm에서 rollout 후 전체 sim state hash가 같다.
  - **Frozen VLA 50 에피소드:** v2 `primary/+y2`와 `distance_4cm/-x4`를 기존 v1 평가(`results/residual_robustness/evals/base_py2.json`, `base_mx4.json`)와 비교했다.

    | 항목 | +y 2cm | −x 4cm |
    |---|---|---|
    | init state hash | 50/50 | 50/50 |
    | 성공 여부 | 50/50 | 50/50 |
    | 에피소드 길이 | 50/50 | 50/50 |
    | 교란 step | 50/50 | 50/50 |
    | 이동 전·후 위치 | 50/50 | 50/50 |
    | 적용 여부 | 50/50 | 50/50 |
    | EEF 거리 | 50/50 | 50/50 |
    | 성공 수 | 34/50 (v1과 같음) | 33/50 (v1과 같음) |

- `info`는 v1 키(`perturb_events`, `perturb_step`, `perturb_dist`, `perturb_axis`, `perturb_sign`)를 그대로 유지한다. 그래서 기존 분석 코드가 그대로 동작한다.
- 기존 `envs/perturbation.py`, `train_residual.py`, `policies/residual_policy.py`, `rl/`은 수정하지 않았다.

## 7. 학습·평가 프로토콜 (공통)

| Split | 세트 / 프리셋 | 조건 | 용도 |
|---|---|---|---|
| `train` | `train_v1` | Early 18cm(v1 규칙), 에피소드마다 ±x/±y 무작위, 2cm, 280 step, 원본 init state | **두 알고리즘 공통 학습 조건**. Residual-SAC `6d_rand_fixa001` run과 방향 난수 stream이 같다. |
| `eval_primary` | `primary` | 0cm, ±x 2cm, ±y 2cm, v1 규칙, 280 step, 50 ep, eval seed 0 | **공식 비교 지표** (M1: 0cm, M2: 2cm 4방향 평균) |
| `eval_secondary` | `distance_4cm`, `timing_early`, `timing_middle`, `timing_late`, `chunk_pre_replan`, `chunk_post_replan`, `chunk_mid` | 학습하지 않은 조건(4cm, Middle, Late, chunk phase). 280 step, 50 ep | 보조 평가. 축별로 따로 보고한다. |
| `eval_ood` | `ood_spatial` | 초기 위치 이동(±2.5cm, BDDL 영역 내) + 0cm/±x/±y 2cm | OOD 옵션. 공식 지표와 섞지 않는다. |
| `diagnostic` | `horizon400`, `validation_timing`, `no_perturbation` | ±y 4cm 400 step 등 | 진단 전용. 비교 지표로 쓰지 않는다. |

- 보고 규칙:
  - 세트별로 따로 보고하고, split이 다른 결과를 평균하거나 합치지 않는다.
  - 모든 평가에서 교란 적용률, 거부 사유, 미발생 사유를 성공률과 함께 보고한다.
  - 미발생 에피소드는 성공률 분모에 포함하고, 별도 비율로도 표시한다.
- Middle과 Late는 정책마다 실제 발생 step과 발생 확률이 다를 수 있다.
  - 정책 간 비교에서는 `info["dynamic_libero"]`의 `step`, `eef_distance`, `status`를 함께 보고한다.
- Checkpoint 선택(validation)은 test와 다른 seed를 쓴다. 기존 Residual 규칙은 eval seed 1000, ±x/±y 2cm × 10 ep였다.

## 8. DSRL-SAC 연동 방법 (사용 예제)

### 8.1 서버 환경

```bash
cd ~/rf/steer-frozen-vla && git fetch && git checkout env/dynamic-libero-v2   # 또는 merge 후 main
source scripts/server_env.sh <gpu>          # $PY, LIBERO 경로, EGL 설정
"$PY" -m pytest -q tests/test_dynamic_libero_v2.py                       # 단위 테스트 (~5초)
DLV2_SIM=1 "$PY" -m pytest -q tests/test_dynamic_libero_v2_sim.py        # 시뮬레이터 테스트 (~3분)
```

### 8.2 학습 환경 만들기

```python
from envs.dynamic_libero_config import load_preset, make_dynamic_libero

cfg = load_preset("train_v1")        # Early 18cm, +-x/+-y random, 2cm, 280 steps (shared training condition)
sim = make_dynamic_libero(cfg)       # LeRobot LiberoEnv + DynamicLiberoV2, one call = one control step
obs, info = sim.reset(seed=episode_seed)          # same seed -> same LIBERO init state order, same direction
obs, reward, terminated, truncated, info = sim.step(action_7d)   # action = executed 7-D VLA action
```

DSRL의 RL action은 noise latent이고, VLA chunk 하나(10 step)마다 한 번 선택된다. 같은 `sim`을 chunk 단위 env로 감싸면 된다. 예시(테스트 `ChunkLatentEnv`와 같은 구조):

```python
class NoiseLatentEnv:                               # policies/noise_policy.py 쪽에 두는 것을 권장
    def __init__(self, sim, vla, n_action_steps=10, max_steps=280):
        self.sim, self.vla, self.n, self.max_steps = sim, vla, n_action_steps, max_steps

    def reset(self, seed=None):
        self.t = 0
        self.vla.reset(seed)
        self.raw_obs, info = self.sim.reset(seed=seed)
        return self.vla.state(self.raw_obs), info  # flat obs, no simulator coordinates

    def step(self, z):                            # z: noise latent chosen by SAC
        self.sim.notify_replan(self.t)            # optional: logs the real re-planning step
        chunk = self.vla.predict_chunk(self.raw_obs, noise=z)   # models/noise_wrapper.py
        ret, terminated, truncated = 0.0, False, False
        for a in chunk[: self.n]:
            self.raw_obs, r, terminated, _, info = self.sim.step(a)
            self.t += 1
            ret += r
            truncated = self.t >= self.max_steps and not terminated
            if terminated or truncated:
                break
        return self.vla.state(self.raw_obs), ret, terminated, truncated, info
```

- `info`는 마지막 step의 값을 그대로 넘기면 된다. `info["dynamic_libero"]`가 에피소드 최종 상태를 담고 있다.
- 280 step 제한은 Residual의 `ResidualEnv`와 같이 env 쪽에서 truncate한다.
- 에피소드 기록과 로그에 남길 키: `dynamic_libero.status`, `reason`, `step`, `eef_distance`, `chunk_phase`, `choice.direction`, `choice.distance_m`.
- Residual-SAC 쪽은 기존 `policies/residual_policy.ResidualEnv`가 그대로 동작한다: `ResidualEnv(make_dynamic_libero(cfg), base, alpha, max_steps=cfg.max_steps)`.
  - 두 wrapper가 같은 seed에서 같은 방향과 크기를 뽑는 것을 시뮬레이터 테스트로 확인했다.

### 8.3 평가

```python
from envs.dynamic_libero_config import load_eval_set, make_dynamic_libero

for cond, cfg in load_eval_set("primary").items():      # "0cm", "+x2", "-x2", "+y2", "-y2"
    sim = make_dynamic_libero(cfg)                      # 280 steps, original init states
    for i in range(cfg.episodes):                       # 50
        obs, info = sim.reset(seed=cfg.episode_seed(i)) # seed i, LIBERO init state i (fresh env per condition)
        ...
```

- 주의: LIBERO는 같은 process 안에서 reset 순서대로 init state를 넘긴다(에피소드 i → init state i).
  - 조건마다 새 env를 만들거나, 같은 env를 쓴다면 조건마다 에피소드 0부터 다시 시작해야 Residual 결과와 짝이 맞는다.
- Frozen Base 기준 rollout과 검증용 기록은 `scripts/run_dynamic_libero_v2.py`로 만든다.

```bash
"$PY" scripts/run_dynamic_libero_v2.py --set primary --condition +y2 --out results/dlv2/base_py2.json
"$PY" scripts/run_dynamic_libero_v2.py --set timing_late --condition=-x2 --out results/dlv2/base_late_mx2.json
"$PY" scripts/summarize_dynamic_libero_v2.py results/dlv2
```

- `-x2`처럼 `-`로 시작하는 조건은 `--condition=-x2` 형식으로 넘긴다.

## 9. 테스트

| 명령 | 결과 |
|---|---|
| `"$PY" -m pytest -q tests` (기존 + v2 단위 테스트) | 87 passed, 11 skipped (시뮬레이터 테스트는 opt-in) |
| └ 기존 테스트 (`test_sac`, `test_perturbation`, `test_residual_env`, `test_training_monitor`, `test_evaluation_summary`) | 23 passed (변경 없음) |
| └ `tests/test_dynamic_libero_v2.py` | 64 passed |
| `DLV2_SIM=1 "$PY" -m pytest -q tests/test_dynamic_libero_v2_sim.py` | 11 passed (약 2분 45초, GPU 1개) |

요구 항목별 근거:

| 요구 항목 | 테스트 |
|---|---|
| 1. Legacy 재현 | `test_legacy_mode_matches_v1_wrapper_exactly` (24 cases), `test_legacy_mode_reproduces_v1_on_the_simulator` (3), Frozen VLA 50 ep × 2 (§6) |
| 2. Early/Middle/Late 구간 | `test_trigger_happens_inside_the_configured_window` (4), `test_timing_windows_trigger_inside_window_on_simulator` (3), §5.1 |
| 3. 충돌·접촉·들림 | `test_robot_contact_blocks_and_is_counted`, `test_lifted_object_is_never_moved`, `test_penetration_rolls_back_and_is_recorded`, `test_workspace_boundary_rejection`, `test_at_most_one_attempt_per_episode` |
| 4. Seed 재현 | `test_direction_and_distance_sampling_is_seed_reproducible_and_covers_all`, `test_direction_stream_matches_residual_random_direction_runs`, spatial 같은 seed → 같은 offset |
| 5. 실제 이동량 | `test_actual_displacement_vector_matches_direction_and_size` (12), §5.1 이동량 = 설정 550/550 |
| 6. 미발생 사유 | `test_window_not_reached_is_reported`, `test_window_skipped_between_steps_is_reported`, `test_chunk_phase_mismatch_reason_when_window_too_short`, `test_info_contract_keys` |
| 7. 공통 인터페이스 | `test_residual_and_dsrl_style_wrappers_share_one_simulator_interface` (ResidualEnv와 chunk 단위 latent env가 같은 sim, 같은 choice) |
| 8. chunk phase | `test_chunk_phase_gate_and_record` (4), `test_chunk_phase_gate_on_simulator`, `test_reported_replans_are_logged_next_to_modular_phase`, §5.3 (550/550) |
| 초기 위치 | `test_spatial_mode_valid_reproducible_and_leaves_libero_init_state_untouched`, `test_target_starts_inside_its_bddl_region_in_world_xy` (50 init states) |
| 설정 | `test_presets_and_eval_sets_load_and_round_trip`, `test_splits_are_disjoint`, `test_invalid_configs_rejected`, `test_no_perturbation_config_is_a_pass_through` |

- 대규모 RL 학습은 하지 않았다.
- 실행한 rollout은 Frozen Base 11개 × 50 에피소드뿐이다. 다른 사용자가 GPU 1~3을 쓰고 있어서 GPU 0 하나로 실행했다.

## 10. 파일

| 파일 | 내용 |
|---|---|
| `envs/dynamic_libero.py` | `DynamicLiberoV2` wrapper, info 키 상수 |
| `envs/dynamic_libero_config.py` | `DynamicLiberoConfig`, `PerturbationSpec`, `SpatialSpec`, `load_preset`, `load_eval_set`, `make_dynamic_libero` |
| `configs/dynamic_libero_v2.yaml` | 학습, 공식, 보조, OOD, 진단 조건 |
| `scripts/run_dynamic_libero_v2.py` | Frozen Base rollout과 검증 기록 |
| `scripts/summarize_dynamic_libero_v2.py` | 적용률, 사유, phase, 실현 가능성, legacy 재현 요약 |
| `tests/test_dynamic_libero_v2.py` | 시뮬레이터 없는 테스트 |
| `tests/test_dynamic_libero_v2_sim.py` | 실제 LIBERO 테스트 (`DLV2_SIM=1`) |
| `envs/perturbation.py` | v1. 변경 없음. |
