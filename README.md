# micro-duck

A small two-legged duck robot, designed digital-first: build it in physics simulation, train its brain there, then build the real one to match.

**Live training page:** each skill's newest brain replayed on the duck's 3D mesh, with test results ([private artifact](https://claude.ai/artifact/DthgE75JEykQcNQ2C4KwzM)).

| Folder | What it is |
|---|---|
| [`body/`](body/REVIEW.md) | The trainable duck: the owner's v3 design (head turns 360°, spring neck, duck-bill mouth) at mid size (0.7×, ~30 cm, 0.8 kg, Dynamixel XL330 servos), built on Open Duck Mini v2 (MuJoCo). Physics checks, the skill training setup, ONNX packaging and the onboard runtime. Start with [`REVIEW.md`](body/REVIEW.md) and [`TRAINING.md`](body/TRAINING.md). |
| [`models/`](models/) | Trained skill brains, one per skill (`mid-<skill>/`): `policy.onnx` for the robot, the training checkpoint, and a model card. |
| [`sim/`](sim/README.md) | Browser 3D sim with hand-built balance physics: sleeps, wakes up, stands up, walks, runs, catches pushes, gets back up. Open `sim/index.html`, works offline. |
| [`LESSONS.md`](LESSONS.md) | Techniques from robots that made it from simulation to a real body. |

## Skills

One brain per skill, all on one shared input layout so the robot switches between them instantly:
stand (idle, looking around, talking) · squat · sit (rests on belly and thighs, the sleep pose) · sit to stand · walk · run · lie down · hop · get up · forward roll.
An aerial flip isn't possible with these servos (about 10× short on leg power), so the trick is a forward roll on the ground.

## Try it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r body/requirements.txt onnx onnxruntime
export MUJOCO_GL=osmesa                      # headless rendering
python body/runtime.py --script "say hello; go to sleep; wake up; walk forward"
python body/train.py --skills --model model/duck_mid.xml --task stand --name mid_stand   # run from body/
python -m unittest body/test_runtime.py
node --test sim/test/*.test.js
```

## Before building the real one

- The v3 parts scaled to 0.7× are drawn around the bigger STS3215 servo; the XL330 (34 × 20 × 26 mm) does not fit the shrunken pockets, so the leg and neck parts need redrawing around it.
- Weigh the printed parts and put the real masses into the model; check the 125° knee fold on a real leg.

## Credits

Body model: [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) by Antoine Pirrone (Apache-2.0), see `body/model/LICENSE` and `body/model/UPSTREAM.md`. XL330 servo parameters from Pollen Robotics' [microduck_rl](https://github.com/pollen-robotics/microduck_rl) (Apache-2.0). three.js (MIT) in `sim/vendor/`.
