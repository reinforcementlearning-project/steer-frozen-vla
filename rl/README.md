# `rl/` — 공통 SAC 학습 프레임워크 (계획서 8-1, 담당: 이다혜)

Residual-SAC(이승연)와 DSRL-SAC(이다혜)가 **같은 SAC 코드**를 사용하도록 만든 모듈이다.
두 방법의 차이는 `ActionSpec`(행동 차원·범위) 하나로만 표현된다.

| 파일 | 내용 |
|---|---|
| `interface.py` | `ActionSpec`(가변 Action 차원/범위), `RLAgent`·`GymLikeEnv` 프로토콜 |
| `sac.py` | `SACConfig`, Tanh-Gaussian Actor, Twin-Q Critic, Target Network, 자동 Entropy 계수, save/load |
| `replay_buffer.py` | FIFO Replay Buffer (npz 저장/복원) |
| `trainer.py` | `TrainerConfig`, `OffPolicyTrainer`: Warm-up, Update 주기/UTD, 로그, 평가, Checkpoint 저장·재시작 |
| `logger.py` | CSV 로거 (`train.csv`, `episodes.csv`, `eval.csv`) |
| `toy_env.py` | 의존성 없는 연속 제어 테스트 환경 `PointReachEnv`, gymnasium 어댑터 |

## 체크리스트 대응

- [x] SAC Actor / Critic / Target Network / Replay Buffer 구성 → `sac.py`, `replay_buffer.py`
- [x] 가변 Action 차원 Policy Interface → `ActionSpec(dim, low, high)`; Residual=7, DSRL=latent_dim
- [x] Update 횟수, Entropy Coefficient, Batch, Replay Warm-up 설정 → `SACConfig`, `TrainerConfig`
- [x] 간단한 연속 제어 환경 학습 테스트 → `scripts/sanity_check_sac.py`, `tests/test_sac.py`
- [x] 학습 Step, Reward, Loss, Checkpoint 저장·재시작 → `OffPolicyTrainer.save_checkpoint()/resume()`
- [ ] 공통 SAC 학습 인터페이스를 이승연과 공유 → 아래 "연결 방법" 전달

## 연결 방법 (이승연의 env wrapper가 지켜야 할 것)

환경은 **gymnasium API**를 따르고 **1-D `np.ndarray` 관측**을 돌려주면 된다.

```python
obs, info = env.reset(seed=...)
obs, reward, terminated, truncated, info = env.step(action)   # action: (ActionSpec.dim,)
# info["success"]   : bool  — Task 성공 (에피소드 동안 OR 처리)
# info["vla_calls"] : int   — 이번 step에서 Frozen VLA forward 횟수 (Budget 기록용)
# 그 외 TrainerConfig.episode_info_keys 에 적은 키 (예: "perturb_step", "perturb_dist")는 episodes.csv에 기록
```

- `terminated`(성공/실패로 끝남)와 `truncated`(최대 Step 도달)를 구분해야 한다. truncated는 bootstrap 한다.
- LIBERO의 dict 관측 → state 벡터 변환(proprio, base action, 인코더 특징 등)은 env/VLA wrapper가 담당한다.

```python
from rl import ActionSpec, SACAgent, SACConfig, OffPolicyTrainer, TrainerConfig

# Residual-SAC: Δa ∈ [-1,1]^7, wrapper 안에서 a_exec = clip(a_base + α·Δa)
spec = ActionSpec(dim=7, low=-1, high=1, name="residual")
# DSRL-SAC:     z ∈ [-b,b]^k,  wrapper 안에서 noise = f(z) → SmolVLA 샘플러
spec = ActionSpec(dim=latent_dim, low=-2.5, high=2.5, name="noise_latent")

agent = SACAgent(obs_dim=env.obs_dim, action_spec=spec, config=SACConfig())
trainer = OffPolicyTrainer(env, agent, TrainerConfig(
    total_env_steps=20_000, warmup_steps=1_000, learning_starts=1_000,
    warmup_policy="zero",          # Residual: 0 = Frozen VLA 그대로 / DSRL: "gaussian" = 원래 N(0,I) 노이즈
    batch_size=256, utd_ratio=1, eval_every=2_000, checkpoint_every=2_000,
    run_dir="results/residual_dyn_s0", seed=0,
    episode_info_keys=["perturb_step", "perturb_dist"],
), eval_env=eval_env)
trainer.train()            # 중단되면: trainer.resume(); trainer.train()
```

평가 전용 재로드: `agent = SACAgent.from_checkpoint("results/.../checkpoints/latest.pt")`

## 출력물

```
<run_dir>/
  config.json                 # trainer + SAC 설정, obs_dim, action_spec
  train.csv                   # env_step, grad_updates, vla_calls, return/success(최근 20ep), losses, alpha, sps, gpu_mem_gb
  episodes.csv                # 에피소드별 return, length, success, vla_calls, + episode_info_keys
  eval.csv                    # 고정 seed(eval_seed+i) 평가: success_rate, return_mean ...
  checkpoints/latest.pt       # agent + optimizer + 카운터 + RNG
  checkpoints/ckpt_XXXXXXXX.pt
  checkpoints/replay_buffer.npz
```

## 주요 설정값

| 설정 | 기본 | 설명 |
|---|---|---|
| `SACConfig.hidden_dims` | [256,256] | Actor/Critic MLP |
| `gamma`, `tau` | 0.99, 0.005 | 할인율, Target Polyak 계수 |
| `init_alpha`, `autotune_alpha` | 0.1, True | Entropy 계수; target entropy = −scale·dim |
| `actor_update_freq` | 1 | Critic n회마다 Actor 1회 (지연 업데이트) |
| `TrainerConfig.warmup_steps` / `warmup_policy` | 1000 / uniform | 초기 무작위 수집 (uniform·gaussian·zero·함수) |
| `learning_starts` | 1000 | 이 크기 이상 쌓이면 업데이트 시작 |
| `update_every`, `utd_ratio` | 1, 1 | N step마다 N·utd 번 Gradient Step |

LIBERO는 Step이 느리므로(VLA 추론 포함) `utd_ratio`를 2–4로 올려 sample efficiency를 높이는 것을 권장한다.
Resume 시 진행 중이던 에피소드는 새로 `reset()`된다 (나머지 학습 상태는 그대로 복원).

## 검증

```bash
pip install -r requirements.txt
python -m pytest tests -q
python scripts/sanity_check_sac.py --env point --steps 15000
python scripts/sanity_check_sac.py --env pendulum --steps 15000
```
