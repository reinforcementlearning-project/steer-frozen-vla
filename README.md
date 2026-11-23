# steer-frozen-vla — Dynamic LIBERO × Frozen SmolVLA RL

**Learning to Correct a Frozen VLA under Dynamic Object Perturbations: Action-Space Residual RL vs. Flow-Noise RL**

동적 물체 교란이 발생하는 LIBERO 환경에서, 가중치를 고정한 SmolVLA의 행동을 소형 SAC 정책으로 보정한다.
보정 위치에 따라 두 방법을 비교한다.

- **Residual-SAC** (이승연): VLA Action에 잔차 보정 `a_exec = clip(a_base + α·tanh(Δa))`
- **DSRL-SAC** (이다혜): Flow-Matching 초기 노이즈를 RL 정책이 선택 `a_exec = SmolVLA(obs, f(z))`

두 방법 모두 같은 SAC 코드(`rl/`)와 같은 Dynamic LIBERO 환경·평가 코드를 사용한다.

## 진행 상황

| 모듈 | 담당 | 상태 |
|---|---|---|
| `rl/` 공통 SAC 학습 프레임워크 (계획서 8-1) | 이다혜 | ✅ 구현·테스트 완료 |
| `models/noise_wrapper.py` SmolVLA 노이즈 주입 (8-2) | 이다혜 | ⬜ |
| `policies/noise_policy.py`, `train_dsrl.py` (8-3) | 이다혜 | ⬜ |
| `envs/libero_env.py`, `envs/perturbation.py` (7-1, 7-2) | 이승연 | ⬜ |
| `models/smolvla_wrapper.py` (7-1) | 이승연 | ⬜ |
| `policies/residual_policy.py`, `train_residual.py` (7-3) | 이승연 | ⬜ |
| `evaluation/evaluate.py` | 이승연 주도 | ⬜ |

## 디렉터리 구조

```
dynamic_vla_rl/
├── rl/                      # 공통 SAC (이다혜) — 자세한 사용법: rl/README.md
│   ├── interface.py         #   ActionSpec(가변 Action 차원), RLAgent / GymLikeEnv 프로토콜
│   ├── sac.py               #   SACAgent, SACConfig
│   ├── replay_buffer.py     #   ReplayBuffer (npz 저장/복원)
│   ├── trainer.py           #   OffPolicyTrainer, TrainerConfig (warm-up, UTD, 로그, 평가, checkpoint/resume)
│   ├── logger.py            #   CSV 로거
│   └── toy_env.py           #   테스트용 PointReachEnv, gymnasium 어댑터
├── scripts/
│   └── sanity_check_sac.py  # 간단한 연속 제어 문제에서 SAC 동작 확인 (Stage 0)
├── tests/
│   └── test_sac.py          # 단위 테스트
├── envs/ models/ policies/ evaluation/ configs/   # (예정)
├── results/                 # 학습 로그·checkpoint (git 제외)
└── requirements.txt
```

## 설치

```bash
pip install -r requirements.txt
# GPU 서버: torch는 CUDA 버전에 맞게 설치 (https://pytorch.org)
```

LIBERO / LeRobot / SmolVLA 의존성은 환경 모듈이 추가될 때 함께 기재한다.

## 공통 SAC 사용법 (요약)

환경은 gymnasium API(`reset() -> (obs, info)`, `step(a) -> (obs, r, terminated, truncated, info)`)를 따르고
1-D `np.ndarray` 관측과 `info["success"]`를 반환하면 된다. 선택적으로 `info["vla_calls"]`를 넣으면 VLA 호출 수가 Budget으로 기록된다.

```python
from rl import ActionSpec, SACAgent, SACConfig, OffPolicyTrainer, TrainerConfig

spec = ActionSpec(dim=7, low=-1, high=1, name="residual")              # Residual-SAC
# spec = ActionSpec(dim=latent_dim, low=-2.5, high=2.5, name="noise_latent")  # DSRL-SAC

agent = SACAgent(obs_dim=env.obs_dim, action_spec=spec, config=SACConfig())
trainer = OffPolicyTrainer(env, agent, TrainerConfig(
    total_env_steps=20_000, warmup_steps=1_000, learning_starts=1_000,
    warmup_policy="zero",      # Residual: "zero"(=Frozen VLA 그대로) / DSRL: "gaussian"(=원래 노이즈)
    utd_ratio=2, eval_every=2_000, checkpoint_every=2_000,
    run_dir="results/residual_dyn_s0", seed=0,
    episode_info_keys=["perturb_step", "perturb_dist"],
), eval_env=eval_env)
trainer.train()                     # 중단 후 재시작: trainer.resume(); trainer.train()

agent = SACAgent.from_checkpoint("results/residual_dyn_s0/checkpoints/latest.pt")  # 평가용 재로드
```

출력: `results/<run>/{config.json, train.csv, episodes.csv, eval.csv, checkpoints/}`
— 설정값·로그 항목 상세는 [rl/README.md](rl/README.md) 참고.

## 실험 공통 규약 (계획서 9장)

- **학습 Budget**: 환경 Step, VLA 호출 수, Optimizer Update 수를 모두 기록 (trainer가 자동 기록)
- **평가**: Task·초기 상태·교란 Seed·Episode 길이·성공 기준 통일 (`eval_seed + i`로 고정 seed 평가)
- **Action Granularity**: Residual(매 제어 Step) vs Noise(매 Chunk) 실행 주기와 상호작용 수 명시
- **보상**: 공통 성공 신호, Shaping 사용 여부와 설정 일치
- 각자 독립적인 LIBERO 환경·평가 코드를 따로 작성하지 않는다.

## 검증 결과

### 공통 SAC (`rl/`, 2026-11-24, CPU / torch 2.15.0, gymnasium 1.4.0)

```bash
python -m pytest tests -q                                   # 12 passed (~5분; -k "not learns" 시 ~25초)
python scripts/sanity_check_sac.py --env pendulum --steps 12000
python scripts/sanity_check_sac.py --env point --dim 7 --steps 20000
```

| 확인 항목 | 결과 |
|---|---|
| 단위 테스트 (가변 차원·범위, log-prob, update, save/load, buffer, resume, warm-up, 학습) | 12/12 통과 |
| Pendulum-v1 평가 return (기준: 무작위 ≈ −1200, 해결 ≈ −200) | 2.4k: −1252 → 4.8k: −126 → 12k: **−109** |
| PointReach 2-D 성공률 (6k step) | ≥ 0.8 (테스트 기준 통과) |
| PointReach 7-D 평가 return (Residual과 같은 Action 차원) | 무작위 −104 → **−25** (성공률 0: 반경 0.1 기준이 7-D에서 매우 엄격) |
| Resume (12k → 13k) | env_step·update 수·buffer(12000) 정상 복원 |
| Checkpoint 재로드 평가 | 학습 직후 평가와 동일 (−121.06 = −121.06) |

## 참고

- Policy Decorator (Residual RL), DSRL (Noise-Space RL), SmolVLA, LIBERO, LeRobot
- Haarnoja et al., *Soft Actor-Critic Algorithms and Applications*, 2018
