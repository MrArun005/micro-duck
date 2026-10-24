# Micro Duck sim

A small two-legged duck robot (about 35 cm tall, 2 kg) that balances with physics in the browser. It sleeps, wakes up, stands up, sits, walks, runs, turns, looks around and dances. Push it and it steps to catch itself. Shove it hard and it falls over, then gets back up.

Open `sim/index.html` through a local web server (see below). Everything it needs is in the repo; three.js comes from `pet/vendor/`.

It's an original design in the spirit of the small expressive bipeds in recent robotics demos: cream egg body, dark neck, round head with a dark visor face, two glowing eyes with eyelids, a small orange bill, two antennas and flat orange duck feet.

## Files

| File | What it does |
| --- | --- |
| `js/duck.js` | Physics, balance controller and behaviours. No graphics, so it runs in Node too (`require('./js/duck.js')`) and can be reused for training or a real robot. |
| `js/render.js` | three.js view. Reads `duck.kinematics()` and `duck.debug` and never changes the simulation. |
| `js/app.js` | Buttons, keyboard, click-to-push and the live readout. |
| `test/` | `node --test` tests. |

## How the physics works

- **Balance.** The body is modelled as a point mass on legs that can change length: the linear inverted pendulum. The feet can only move the centre of pressure (ZMP) around inside the soles, and the body accelerates away from it: `a = ω²(c − p)`, with `ω = √(g/h)`.
- **Capture point.** `ξ = c + v/ω` is where a foot has to land for the duck to come to a stop. While ξ stays inside the feet, the ankles can hold it. Once ξ leaves the feet, the duck has to step, and the swing foot is re-aimed at the predicted ξ on every tick while it's in the air. If ξ runs further than a leg can reach, the duck falls.
- **Walking and running** use the same controller. Each foot lands a little behind the predicted capture point, so the body keeps rolling forward at the target speed. Walking uses 0.25 s steps and running uses 0.18 s steps with a lower crouch. The duck keeps the short running steps until it has slowed down, and it turns more gently the faster it goes. Both rules exist so that turning out of a run can't ask for a step longer than the legs.
- **Falling** is a rigid-body simulation: a 2 kg body with inertia, contact points on the shell, head and feet, spring-damper ground contact, and Coulomb friction.
- **Getting up** applies a righting torque with a 4 N·m limit, and tucks the feet in while doing it. If it can't get up, it rests and tries again.
- **Joints** behave like serial-bus servos: stiff springs with a speed limit. Eyelids, eye glow and antennas are joints too.

Be aware of what is simplified. The get-up is a single limited torque on the body, not individual leg motors pushing on the ground. The walk is a model-based controller (capture-point stepping), not a learned policy, so it's tidier than a real robot and doesn't model motor backlash, foot slip while standing, or sensor noise.

## Controls

- **Buttons:** Wake up, Sleep, Stand up, Sit, Walk, Run, Turn left/right, Stop, Look around, Dance, Push, Big shove, Physics overlay.
- **Keyboard:** `W A S D` or the arrow keys drive. Hold `Shift` to run. `Space` puts it to sleep or wakes it up.
- **Mouse/touch:** drag to orbit, scroll or pinch to zoom, double-click to reset the view. Click the duck to push it away from the camera.
- **Physics overlay:** shows the support area (green), the capture point (orange), the centre of pressure (blue) and the centre of mass dropped to the floor.

## Run and test

```bash
python3 -m http.server          # from the repo root, then open http://localhost:8000/sim/
node --test sim/test/*.test.js
```

The tests drive the simulation with fixed 1/240 s steps. Each test checks one of these:

- it stands for 5 s
- it catches small and firmer pushes from all sides
- a 1.5 m/s shove knocks it over, and it's standing again within 15 s
- 3 s of walking covers more than 0.4 m
- running is faster than walking
- it turns 90° on the spot
- sleep → wake → stand reaches standing idle
- turning out of a full-speed run doesn't fall
- a mixed keyboard-driving sequence doesn't fall
- a 60 s walk doesn't drift into a fall
- the legs keep their segment lengths

## Next step

Build a MuJoCo model of this body and train a walking and get-up policy with reinforcement learning, in [`body/`](../body/TRAINING.md). `duck.js` stays the reference for behaviours and the browser preview. The trained policy would replace the capture-point controller and the simplified get-up torque.
