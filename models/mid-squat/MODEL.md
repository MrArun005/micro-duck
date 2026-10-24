# mid-squat

Skill brain **squat** for the mid duck (`body/model/duck_mid.xml`), trained with PPO in MuJoCo
(`body/train.py --skills --task squat`). Command it was trained under: `sit`.
Packaged 2026-10-24.

- Training steps: **1,500,000** (about 8.3 h of simulated duck time)
- Test score: 100% success, mean reward 152.3 (train.py exam)
- ONNX vs stable-baselines3 on 1000 observations from SkillEnv rollouts: max abs
  difference **4.2e-07**

## Files

| File | What |
|---|---|
| `policy.onnx` | The brain. Input `obs` (N×60 **raw** values), output `action` (N×11, each −1…1). Normalizer built in. |
| `model.json` | The same facts, machine-readable (used by `body/runtime.py`) |
| `policy.zip` + `vecnormalize.pkl` | To continue training |

## Every 1/50 s (50 Hz)

1. Build the 60 inputs below from the sensors (no normalization, no filtering).
2. `action = brain(obs)`.
3. Servo targets = home; policy servos += action_scale × action; head_yaw / jaw from the personality
   layer (look, talk); clip to the servo range. No smoothing: the brain was trained without it.
4. Remember `action` for the next observation's `last_action`.

## Observation (raw)

| # | Name | Size | Meaning |
|---|---|---|---|
| 0 | gravity_in_imu_frame | 3 | unit vector, from the IMU orientation |
| 1 | gyro_x0.25 | 3 | IMU angular velocity, rad/s × 0.25 |
| 2 | accelerometer_div_9.81 | 3 | IMU linear acceleration, m/s² / 9.81 |
| 3 | joint_pos_minus_home | 13 | every servo, rad, minus the home pose |
| 4 | joint_vel_x0.05 | 13 | every servo, rad/s × 0.05 |
| 5 | last_action | 11 | the brain's previous output, after clipping to ±1 |
| 6 | command_one_hot | 7 | [stand, walk, run, sit, lie, jump, roll] |
| 7 | velocity_command | 3 | vx, vy (m/s), yaw rate (rad/s); zeros unless walk/run |
| 8 | gait_clock | 2 | sin, cos of 2π·f·t; zeros unless walk (1.8 Hz) / run (2.6 Hz) |
| 9 | foot_switches | 2 | left, right sole on the floor (0/1) |

## Servos (observation order; the action covers the policy servos in this order, skipping external ones)

| # | Servo | Home (rad) | Range (rad) | Action scale (rad per unit) |
|---|---|---|---|---|
| 0 | left_hip_yaw | +0.002 | -0.524 … +0.524 | 0.4 |
| 1 | left_hip_roll | +0.053 | -0.436 … +0.436 | 0.4 |
| 2 | left_hip_pitch | -0.630 | -1.222 … +0.524 | 1.0 |
| 3 | left_knee | +1.368 | -1.571 … +2.182 | 1.0 |
| 4 | left_ankle | -0.784 | -1.571 … +1.571 | 1.0 |
| 5 | head_pitch | +0.000 | -0.785 … +0.785 | 0.6 |
| 6 | head_yaw | +0.000 | -3.142 … +3.142 | external |
| 7 | right_hip_yaw | -0.003 | -0.524 … +0.524 | 0.4 |
| 8 | right_hip_roll | -0.065 | -0.436 … +0.436 | 0.4 |
| 9 | right_hip_pitch | +0.635 | -0.524 … +1.222 | 1.0 |
| 10 | right_knee | +1.379 | -1.571 … +2.182 | 1.0 |
| 11 | right_ankle | -0.796 | -1.571 … +1.571 | 1.0 |
| 12 | jaw | +0.000 | +0.000 … +0.610 | external |

```python
import onnxruntime as ort
brain = ort.InferenceSession("policy.onnx")
action = brain.run(None, {"obs": obs[None].astype("float32")})[0][0]
```
