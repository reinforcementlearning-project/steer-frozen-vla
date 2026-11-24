# Day 1 실행 방법

계획·역할분담: [최신 노션 문서](https://www.notion.so/42178ef6f26881f29c64c7a8abb7b108).
최상위 README와 `rl/README.md`의 공통 SAC 규약을 따른다. 공통 `rl/` 구현은 변경하지 않는다.

## 현재 서버에서 실행

작업 디렉터리는 `~/rf/steer-frozen-vla`. 아래 환경 설정은 기존 runtime과 라이브러리를 읽어서 사용하며 별도 환경 설치 도구가 아니다. 기존 연구 폴더에 코드나 패키지를 쓰지 않는다.

```bash
cd ~/rf/steer-frozen-vla
source scripts/server_env.sh 0
# 최초 1회. 이미 configs/libero/config.yaml이 있으면 덮어쓰지 않고 중단한다.
"$PY" -B scripts/setup_libero_paths.py
# 최초 1회. 기존 환경 대신 rf/python-deps에만 보충한다.
"$PY" -B -m pip install --target ~/rf/python-deps num2words==0.5.14 docopt==0.6.2
"$PY" -B scripts/prepare_smolvla.py
```

체크포인트와 backbone revision은 다운로드 코드에 고정되어 있다. 큰 모델 파일은 `~/rf/.cache/huggingface`에 보관하고 Git에 넣지 않는다. 첫 환경 실행에서 hf-libero가 `~/.cache/libero/assets`를 자동 생성한 이력이 있다. 이후 adapter는 process-local assets cache를 설정해 기존 `RF_BASE/.cache/libero/assets`를 재사용한다. 기존 연구 파일은 변경하지 않았다.

## 실제 Frozen Base rollout 및 chunk 확인

```bash
"$PY" -B scripts/smoke_rollout.py --n-action-steps 1 --out results/base1.json
"$PY" -B scripts/smoke_rollout.py --n-action-steps 50 --max-steps 55 --out results/chunk50.json
```

checkpoint 기본 `n_action_steps=50`과 실제 실행 옵션을 구분한다. generator는 50개 action을 생성하지만 queue에서 실행하는 수는 옵션으로 결정한다. `n_action_steps=1`은 매 제어 step 재추론, 10은 10 step마다 재추론한다. simulator 제어 주기는 20Hz이며 wall-clock 실시간 실행 보장은 아니다.

## Dynamic LIBERO 평가

```bash
"$PY" -B scripts/smoke_rollout.py --n-action-steps 10 --episodes 50 \
  --target alphabet_soup_1 --distance-cm 2 --out results/eval50_2cm.json
# 0 / 2 / 4cm 각각 별도 실행. GPU 선택은 source scripts/server_env.sh <index>.
"$PY" -B -m evaluation.summarize results/eval50_0cm.json results/eval50_2cm.json results/eval50_4cm.json
```

Task: `libero_object`, task ID 0, “pick up the alphabet soup and place it in the basket”.
최초 10 step 이후 EEF-대상 거리 5–18cm, 로봇 접촉 없음, 대상이 초기 높이에서 1.5cm 이내일 때 y+ 방향으로 episode당 1회 이동한다. Table workspace 및 신규 3mm 초과 침투/로봇 접촉을 검사하고 무효 이동은 원복한다. 각 이벤트의 실제 이동량·시점·거절 사유를 남긴다. 0cm는 같은 gate의 sham 이벤트이며 속도를 보존한다. VLA가 다음 관측을 받을 때 새 카메라 프레임을 사용하되 이미 queue에 든 action은 예정대로 실행한다.

비교 조건은 같은 init state ID와 noise seed를 사용한다. JSON의 초기 simulator state hash로 짝이 일치하는지 검증한다. 물체 정답 좌표는 이벤트를 정의할 때만 쓰고 정책 관측에는 넣지 않는다. 거리·시간 기준 gate는 모든 파지 상태를 완벽히 판정하는 보장은 아니므로 새 task에는 재검증이 필요하다.

## Residual-SAC 연결 검증

관측은 15차원 float32: proprioception 8 + 역정규화된 base action 7.
SAC의 7차원 출력은 이미 tanh로 [-1,1]에 제한되어 있으므로 `clip(base + alpha*u, -1, 1)`을 쓴다. 기본 alpha=0.1, gripper 잔차는 0으로 둔다(기본 gripper action 보존). 즉 7차원 중 마지막 잔차는 현재 사용하지 않는다. clipping 기준의 zero-residual 복원을 검증한다.

```bash
# 실제 LIBERO transition 32개, 공유 Trainer, zero warm-up, SAC update 1회
"$PY" -B train_residual.py --total-env-steps 32 --warmup-steps 32 --batch-size 32 \
  --run-dir results/trainer_smoke
# checkpoint, optimizer, buffer, RNG를 복원하고 1 step 더 진행
"$PY" -B train_residual.py --total-env-steps 33 --warmup-steps 32 --batch-size 32 \
  --run-dir results/trainer_smoke --resume
```

`vla_calls`는 공통 Trainer가 무시하는 reset info 대신 첫 control step에 초기 추론 비용을 합산한다. 다음 state의 base action 준비에 든 추론도 포함한다. Time-limit 종료는 truncated이며 bootstrap할 수 있도록 실제 다음 base action을 유지한다. 성공 종료는 terminated이다. 재시작은 진행 중 simulator 상태를 복원하는 대신 새 episode로 시작한다.

이 명령은 **연결·업데이트 검증**이다. 성능이 개선된 Residual 정책이나 본 학습 완료를 뜻하지 않는다. 본 학습은 교란 조건·보상·예산 확정 후 독립 seed 최소 2개(목표 3개)로 진행하고 조건별 50 episode 이상 및 95% CI로 평가한다. 학습 곡선의 x축은 환경 step으로 통일한다. 이 진입점은 학습 중 평가를 자동 수행하지 않으며 별도 평가가 필요하다.
