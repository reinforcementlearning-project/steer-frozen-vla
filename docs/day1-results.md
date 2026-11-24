# Day 1 검증 결과 — 2026-11-24

요청한 네 항목의 기반 검증을 완료했다. Residual-SAC 본 학습이나 성능 개선을 입증한 결과는 아니다.

## 1. SmolVLA + LIBERO 실행

- 체크포인트: [lerobot/smolvla_libero](https://huggingface.co/lerobot/smolvla_libero), revision `a24f1f380461b665158d52acd140152c79e8b36d`.
- Task: LIBERO-Object ID 0, alphabet soup → basket.
- Frozen checkpoint 로딩, MEAN_STD 전·후처리 로딩, 실제 Action 생성 및 Episode 실행 확인.
- `n_action_steps=1`: 205 control step / 205 VLA 호출 후 성공(reward=1).
- 이 1회 실행은 동작 확인이며 단독 성공률 추정치로 사용하지 않는다.
- [실제 로그](evidence/base1.json), [환경 버전·소스 해시](evidence/environment.json).

## 2. Action Chunk·정규화·주기

| 항목 | 확인 결과 |
| --- | --- |
| checkpoint 기본 실행 길이 | `n_action_steps=50` |
| 생성 chunk | `(1, 50, 7)` |
| 초기 Gaussian noise | `(1, 50, 32)` |
| 환경에 넘긴 Action | `(7,)`, 최종 `[-1,1]` clipping |
| proprioception | EEF 위치 3 + axis-angle 3 + gripper qpos 2 = 8 |
| 카메라 | agentview / eye-in-hand → image/image2 → camera1/camera2; H/W 180° 회전 |
| 정규화 | checkpoint 통계로 state·action MEAN_STD; action 역정규화 후 residual 적용 |
| simulator 제어 주기 | 20Hz, wall-clock 실시간 보장 아님 |
| 실행 길이 50 | 55 step에서 VLA 2회 호출(0, 50 step에서 추론) |
| 실행 길이 1 | 매 control step 재추론 |
| 아래 교란 비교 | 실행 길이 10, 10 control step마다 재추론 |

[길이 50 실측 로그](evidence/chunk50.json). A10에서 정책 추론 평균 약 0.26초/호출,
PyTorch peak allocated VRAM 약 1.18GiB. GPU 전체 메모리 점유나 12GB GPU 실측과는 다르다.

## 3. Dynamic LIBERO 교란·Base 성공률

Gaussian random noise Frozen Base, 같은 checkpoint/task, 같은 초기 상태 ID 0–49와 같은 noise seed 0–49.
`n_action_steps=10`, episode 최대 280 control step, y+ 이동. 0cm도 동일한 gate의 sham 이벤트를 사용했다.

| 이동 거리 | 성공 / 평가 Episode | 성공률 | Wilson 95% CI |
| --- | --- | --- | --- |
| 0cm | 47 / 50 | 94% | 83.8–97.9% |
| 2cm | 34 / 50 | 68% | 54.2–79.2% |
| 4cm | 12 / 50 | 24% | 14.3–37.4% |

초기 simulator state hash로 **서로 다른 50개 초기 상태** 및 조건 간 동일한 짝을 검증했다.
세 조건 각각 50/50 episode에서 이벤트가 적용됐고 실제 이동량은 0/0.02/0.04m와 일치했다.
이벤트 직전 robot contact가 없고 이동 시 z 높이를 보존했다. 신규 침투·로봇 접촉을 검사하는 거절/원복 경로도 테스트했다.

- [0cm 원시 로그](evidence/eval50_0cm.json)
- [2cm 원시 로그](evidence/eval50_2cm.json)
- [4cm 원시 로그](evidence/eval50_4cm.json)
- [통계 요약](evidence/eval50_summary.json), [짝·교란량 검증](evidence/paired_validation.json)

본 task와 실행 주기에서는 2/4cm가 Base 성능을 낮추면서도 성공이 가능한 조건이었다.
2cm를 초기 학습 후보로, 4cm를 더 어려운 평가 후보로 검토할 수 있다. 아직 학습/held-out 분할을 확정한 것은 아니다.
위 CI는 episode 수준 불확실성이며 독립 RL 학습 seed 반복 결과가 아니다. 다른 task·미학습 교란 일반화와 고정 `w=0` 비교는 아직 수행하지 않았다.

## 4. Residual-SAC 연결·업데이트

다혜의 최상위 README와 `rl/README.md`를 읽고 공통 `rl/` 소스 수정 없이 연결했다.

- 관측 `(15,)` float32: proprioception 8 + 역정규화 base action 7. Oracle 물체 좌표 입력 없음.
- 공통 `ActionSpec(7,-1,1)` 사용. SAC가 이미 tanh를 적용하므로 wrapper에서 이중 적용하지 않음.
- 기본 gripper residual은 0. Zero-residual, Action 범위, transition 정렬, truncated bootstrap 검증.
- 실제 LIBERO transition 32개 → **공통 OffPolicyTrainer로 Actor·Critic 1회 업데이트**.
- 32 step / 1 update / VLA 호출 33회 상태 저장 후 resume → 33 step / 누적 2 updates / VLA 호출 35회.
- Actor·Critic parameter 변경 확인, VLA 모든 parameter frozen·gradient 없음 확인.
- 별도 agent checkpoint 저장·재로드의 deterministic action 일치 확인.

[Trainer 결과](evidence/trainer_summary.json), [resume 결과](evidence/trainer_resume_summary.json),
[agent 재로드 검증](evidence/residual_update.json).

테스트: `python -m pytest tests -q -k "not learns"` → **21 passed, 1 deselected**.
제외한 항목은 기존 장시간 학습 테스트이며, 본 검증으로 정책의 학습 성능이나 수렴을 주장하지 않는다.

## 다음 단계

공동으로 실행 길이·보상·교란 조건·학습 예산을 확정하고, Dynamic LIBERO에서 Residual-SAC 본 학습을 진행한다.
독립 학습 seed 최소 2개(목표 3개), 정상/학습 교란/미학습 교란의 조건별 평가 50 episode 이상과 95% CI가 필요하다.
이다혜 담당 DSRL·공통 SAC와 동일 환경·checkpoint·평가 규약을 공유하며, random-noise Base와 고정 `w=0`을 별도 대조군으로 유지한다.

재실행 명령과 환경 분리 방식: [Day 1 실행 방법](day1-runbook.md).
