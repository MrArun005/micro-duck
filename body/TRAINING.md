# Training the duck body

Plan for teaching the body in `model/duck.xml` four skills: **stand** (and survive pushes),
**sit-to-stand**, **get up from lying**, **walk**. The environment is ready and smoke-tested;
no long run has been started.

## Setup

```bash
cd micro-duck
python3 -m venv .venv && . .venv/bin/activate
pip install -r body/requirements.txt          # train/requirements.txt + pyyaml
apt-get install -y libosmesa6 && export MUJOCO_GL=osmesa   # offscreen renders only

python body/build_model.py     # tweaks.yaml -> model/duck.xml (re-run after every tweak)
python body/inspect_body.py    # specs
python body/checks.py          # stand-hold, crouch, self-collision, lying checks
python body/render.py          # renders/*.png
python body/train.py --quick   # smoke test (~10 s)
```

## The environment (`env.py`)

- **50 Hz control**, 0.002 s physics. Action = 14 joint-target offsets from the home pose
  (legs ±1.0 rad on pitch joints, ±0.4 on hip yaw/roll; head 0.4–1.0), clipped to the servo range.
- **One observation layout for every skill** (61 numbers, LESSONS §3): gravity in the IMU frame,
  gyro, accelerometer, joint angles and velocities, last action, command one-hot
  `[stand, sit, walk]`, walk velocity command, gait clock, two foot switches. Unused inputs are
  zeros. A `sit` command slot is already reserved so a sleep/sit skill can be added without
  changing the layout (it needs the knee tweak, REVIEW.md #6).
- **Rewards are products** (upright × height × movement × pose), so standing still earns nothing
  under a walk command and lying earns nothing under a stand command. Getting up and sit-to-stand
  track a reference that rises at a fixed rate (10 cm/s height, 0.8/s uprightness): no jackpot for
  getting there violently, no pay for staying down. Smoothness penalties are off until
  `--smooth-after` (introduce them only once the skill exists).
- **Domain randomization, fresh every reset** (never accumulates): body masses ±10% (inertia
  scaled), trunk CoM ±1 cm, servo kp ±15%, **torque limit 1.91–3.23 N·m** (7.4 V stall … upstream),
  joint damping ±20%, friction loss ±30%, floor friction 0.4–1.0, 0–20 ms command delay, ±1° IMU
  mounting error, sensor noise, random shoves while standing and walking.
- **Exam** (`success`): upright (> 0.9) at standing height for the stand tasks; for walk also a
  running velocity-tracking error under 0.06.
- Episode lengths: stand/walk 10 s, sit-to-stand 5 s, get-up 6 s. Stand and walk end on a fall.

## Smoke test (done)

`python body/train.py --quick`: 4,096 PPO steps on 2 environments in about 4 s:

| task | success | mean return | meaning |
|---|---|---|---|
| stand | 1/1 | 493 | servos holding home already stand, even with shoves |
| sit_to_stand | 1/1 | 245 | trivial with today's 90° knees: the crouch is only 10 mm deep |
| get_up | 0/1 | 22 | untrained: stays down and earns almost nothing (as designed) |
| walk | 0/1 | 30 | untrained: doesn't track the velocity command |

Environment throughput on this container (4 CPU cores): **~2,000 env steps/s with 4 processes,
~2,400 with 8** (random actions, contacts included). PPO updates cost roughly another 20–30%.

## Plan

Train one skill at a time, then all four mixed, keeping every checkpoint and picking by exam
(LESSONS §6). Commands are run from the repo root.

| stage | command | steps | CPU here (≈1.7k steps/s) | one GPU (MJX / mjlab port) |
|---|---|---|---|---|
| 1 stand + pushes | `python body/train.py --task stand --steps 5000000 --name stand` | 5 M | ~50 min | minutes |
| 2 walk | `python body/train.py --task walk --steps 40000000 --smooth-after 25000000 --name walk` | 30–50 M | 5–8 h | ~15–30 min |
| 3 get up | `python body/train.py --task get_up --steps 80000000 --smooth-after 50000000 --name getup` | 50–150 M | 8–25 h | ~30–60 min |
| 4 sit-to-stand | `python body/train.py --task sit_to_stand --steps 10000000 --name sit` | 10 M | ~2 h | minutes |
| 5 all four, mixed | `python body/train.py --steps 60000000 --smooth-after 30000000 --resume body/runs/walk/policy.zip --name duck` | 60 M | ~10 h | ~1 h |

```bash
tensorboard --logdir body/runs                          # watch the reward climb
python body/train.py --resume body/runs/walk/checkpoints/policy_20000000_steps.zip --name walk   # after a crash
```

Notes on compute:

- Stand and sit-to-stand are cheap and fine on this CPU. Walking took Open Duck's own setup
  hundreds of millions of GPU steps with a reference-motion imitation reward; our walk reward is
  plain velocity tracking, which is simpler but slower to learn. Get-up is the hardest (Microduck
  needed curricula and starts near the end of the motion). Expect stages 2 and 3 to need a GPU:
  MuJoCo Playground (MJX) or mjlab runs thousands of these robots in parallel; a single RTX 4090
  does roughly 100k–300k steps/s on a model this size, versus ~2k here.
- Porting to a GPU means re-writing `env.py`'s step and reward in JAX (MJX) or using mjlab's task
  API; `model/duck.xml` loads unchanged in both. Keep the observation layout identical.
- Do the decisions in REVIEW.md first (knee range, torque limit, measured masses): a policy trained
  on the wrong body is wasted compute.

## Before a policy goes on the robot

1. Export to ONNX **with the observation normalizer baked in** and check it against the original
   (adapt `train/package.py`: swap `BoltEnv` for `DuckEnv`, keep the check).
2. Package the run into `models/<name>/` with its training data (HANDOFF.md rule).
3. Run it on the real duck with exactly the same 50 Hz loop, action scale, clipping and no extra
   filtering (or the same filtering in sim). Calibrate the IMU on the robot; randomization does not
   fix a fixed mounting offset.
4. Fall and get-up tests on hardware only after the TPU bumpers are on (REVIEW.md #8).
