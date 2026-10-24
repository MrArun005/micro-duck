# Lessons from robots that already work

Techniques collected from open-source robots that made the jump from simulation to a real
body, in our own words, so we can learn from them and do better rather than copy them.
Main sources: Pollen Robotics' Microduck training notes
([microduck_rl](https://github.com/pollen-robotics/microduck_rl), Apache-2.0), Open Duck Mini's
sim-to-real guide ([Open_Duck_Mini](https://github.com/apirrone/Open_Duck_Mini), Apache-2.0),
Petoi Bittle/Nybble, Sesame, and our own training of Bolt.

## 1. The simulation must match the body
- **Weigh what you'll actually build.** Printed parts with infill weigh less than solid CAD
  parts; use the slicer's mass estimate for every part in the simulation model.
- **Measure the motors.** Feed the simulator the real motor's stiffness, damping, friction,
  armature and torque limit (Open Duck uses the BAM tool for this). If simulated motors behave
  differently from real ones, the trained brain fails or moves chaotically on the robot.
- **Model the slop.** Cheap gears have backlash (a couple of degrees of play); simulate it.
- **Check the physics before training.** A target pose must actually be stable: hold it for a
  few seconds from slightly random starts and check the tilt, not just the height. Measure target
  heights from the simulated robot; never guess them. (Microduck lost days to a target 5 mm off.)

## 2. Make the brain robust (domain randomization)
- Vary mass, friction, motor strength, sensor noise, sensor mounting angle and control delay
  every episode, so the brain learns the skill and not one exact simulation.
- Randomization must **reset** every episode, never pile up; a randomizer that slowly drifted
  the centre of mass silently ruined Microduck runs for months.
- Randomization teaches tolerance to random error, not to a fixed offset: a crooked IMU must be
  calibrated on the robot.

## 3. Training setup that transfers
- Control at **50 Hz**. Train with PPO. Export to **ONNX** and run it on the robot's computer.
- **Bake the observation normalizer into the exported model.** Simulation playback applies it
  anyway and hides the bug; a hand-converted model without it fails on hardware.
  (Our `train/package.py` does this and checks the export against the original.)
- Whatever filtering or smoothing the brain is trained with, the robot must use exactly the
  same, and vice versa.
- Keep one input layout shared by all skills (walk, sit, trick) so the robot can swap brains
  instantly; skills that don't use an input get zeros, but the input stays.
- Always run a tiny smoke test (a few iterations) before a long run.

## 4. Reward design: the hard-won rules
- **RL optimizes exactly what you wrote, not what you meant.** Any freedom left open gets
  exploited (whipping instead of rolling, propping up on the head instead of standing). Define
  the trick with hard checks on contacts and orientation, not small nudges.
- **No jackpots.** A reward paid every step once a goal is reached pays the robot to get there
  as violently as possible. Track a target that moves at a fixed rate instead.
- **Never reward being in a bad state** (fallen, low): the robot will lie there and farm it.
  Reward *progress* (getting more upright pays, staying flat pays nothing).
- **Adding rewards together creates compromises.** If leaning halfway earns 80% of every term,
  the robot leans. Multiplying the terms makes any single failure cost everything.
  *This is likely why our Bolt dog never learned sit and lie: standing still already earned most
  of the sum under every command.*
- **Introduce smoothness penalties after the skill exists.** Early on, any cost for trying makes
  "do nothing" the winning strategy.
- **Check the sign of every penalty.** A penalty with the wrong sign becomes a reward for the
  bad behaviour.
- **Gaussian tracking width ≈ the error you still care about.** Too loose and there's no signal;
  too tight on something unavoidable (a heavy head must bob while walking) and the robot freezes.
- **Train the idle command explicitly** (all zeros = "stand still"); random sampling almost never
  produces it. Rare but important commands (turn in place) need their own share of practice.

## 5. Curricula
- Make things harder only after the current skill has consolidated. If a metric drops exactly
  when a stage starts, stretch the stages.
- To learn the last part of a trick, start some episodes near the end of it, so the hard part
  gets practice.
- Resuming the same task keeps its curriculum progress; starting a new task from another task's
  brain should restart the curriculum.

## 6. Reading a run
- Total reward can rise from side terms while the actual trick never happens: watch the main
  task term itself.
- **Measure before theorizing.** When a run "fails", test the actual checkpoint headless (exam,
  end-state clusters) before changing rewards. Several "failures" were really early checkpoints
  or a bad success test.
- **Watch the video.** Numbers can pass while the motion looks wrong.
- Report what the rollouts show ("rolls but face-plants 1 in 3"), not "it works".

## 7. Hardware choices that small robots converge on
- **Smart serial bus servos** (Feetech STS3215, Dynamixel XL330) over plain hobby servos: one wire
  for all motors, position and load feedback, tunable stiffness. That feedback is what the brain
  observes.
- Small computer in the robot (Raspberry Pi Zero 2 W / CM-class, Rockchip boards) running the
  ONNX brain at 50 Hz; heavy thinking (speech, language model) can run elsewhere over Wi-Fi.
- An **IMU** for tilt and spin is essential; foot or contact switches help.
- A **screen or LED eyes** is the cheapest way to make a robot feel alive (Sesame, Microduck,
  Agile ONE).
- Frames: 3D-printed PLA, laser-cut wood that slots together (Nybble), or interlocking plastic
  (Bittle); TPU for grippy feet. Keep weight low: it decides what the motors can do.
- **Physics sets the pace.** A 25 cm robot naturally tumbles at 3.5–5.5 rad/s; don't cap speeds
  with human intuitions, punish impacts and thrashing instead.

## 8. Our own experience so far (Bolt dog, `models/bolt-dog-3M`)
- 4 CPU cores trained stand and walk in ~3M steps (~1 h); sit and lie never emerged (see the
  reward-composition lesson above).
- Learning wobbles: walk was 100% at 3M, 80% with more falls at 3.5M. Keep every checkpoint and
  pick by exam, not by "latest".
- Filming and testing every checkpoint (`train/watch.py`) made progress and regressions obvious.
