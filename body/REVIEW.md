# Duck body review

A hardware-engineer pass over the Open Duck Mini v2 simulation model before we train on it.
Model: Open Duck Playground `open_duck_mini_v2.xml` (the one the real ducks' walking policies were
trained on; see `model/UPSTREAM.md`). Built model: `model/duck.xml` = upstream + `tweaks.yaml`.

Every number below comes from `inspect_body.py` / `checks.py` (outputs in `inspect.txt`,
`checks.txt`, `checks_upstream.txt`, `specs.json`, `checks.json`). Re-run them after any tweak.

## Specs (built model)

| | Model | Real build (duck/README, build/) | Verdict |
|---|---|---|---|
| Mass | **2.107 kg** (CAD inertials) | ~2 kg, "simulation model 2.06 kg" | within 2.5%; weigh the real one |
| Joints | 14 servos: 5 per leg (hip yaw, roll, pitch, knee, ankle) + neck pitch, head pitch, yaw, roll | 14 × STS3215 + 2 SG90 antenna servos | match (antennas are fixed mass in sim) |
| Height | head shell top 430 mm legs straight, 395 mm in the home stance (483 mm to antenna tips) | "about 42 cm with legs extended" | match |
| Leg | thigh 78.7 mm, shin 78.7 mm, ankle→sole 38.8 mm, hip-pitch axis→sole 196 mm straight | | |
| Foot | 103 × 41 mm TPU sole, 8 mm | foot_bottom_tpu 102 × 41 mm | match |
| Stance | hips 70 mm apart, soles at y ±81 mm (support 190 mm wide, 92 mm long) | | |
| Home stance | base 161 mm, CoM height 205 mm, knees 78° | | |
| CoM | x = −31 mm (hip axes at −19 mm), 15 mm behind the foot centre; 31 mm from the heel edge, 61 mm from the toe | | backward is the short side |
| Head | 0.41 kg (19% of the robot) with its CoM at 363 mm | | top-heavy, see 8 |
| Servo model | position servo kp 13.37 N·m/rad, kv 0, joint damping 0.56, friction loss 0.068 N·m, armature 0.027 kg·m², limit ±3.23 N·m | STS3215 7.4 V | BAM-identified upstream; torque limit questionable, see 5 |
| Standing holding torque | knees 0.44–0.50 N·m, everything else < 0.2 N·m | stall 1.91 N·m (19.5 kg·cm) | 26% of stall at the knees: fine |

## Stand-hold test (LESSONS.md §1)

Home pose held by the servos for 5 s, 10 trials, each starting with σ = 0.05 rad joint error and
up to ±3° random body tilt:

- **0 falls.** Max tilt during the hold 4.35° (from the initial ±3° plus landing), end tilt ≤ 0.31°.
- Height settles within 1.1 mm of the reference; drift over the last 4 s ≤ 1.7 mm.
- The body shifts up to 23 mm while the start error settles (one foot lands first), then
  stays put: 0.6 mm of slide over the last 4 s.
- Same result with 1 solver iteration (upstream, GPU-tuned) and 100 iterations, so the cheap
  setting is fine for standing.

The pose is stable. Target heights for training (stand 161 mm base, 249 mm trunk site) are
measured from this settled pose, not guessed.

## Issues and tweaks

Status: **APPLIED** = on by default in `tweaks.yaml` section A. **OPTIONAL** = in section B,
off until you decide.

### 1. Only the soles can touch anything (APPLIED)
**What:** upstream exports every part as visual-only except the two TPU soles. Trunk, head, thighs,
shins and feet pass through the floor and through each other.
**Evidence:** dropped on its back, front or side, the upstream robot ends up hanging upside down
from its feet with the trunk 192 mm *below* the floor (`checks_upstream.txt`).
**Why it matters:** sitting, lying, falling and getting up cannot be trained at all; a walking
policy can also learn to cross its legs through each other.
**Tweak:** the real printed shells become colliders (MuJoCo uses each part's convex hull): trunk
shells, head shell + antenna holders, thigh covers, shin plates, foot tops. 22 hulls, zero added
mass. Legs collide with each other and the floor; head collides with trunk and floor. Now the robot
rests on its back (trunk CoM 106 mm), front (116 mm) or side.

### 2. Joint limits are rigid stops right at the servo's command limit (APPLIED)
**What:** upstream sets the hard joint range equal to the command range.
**Evidence:** upstream model, crouched onto the 90° knee limit: the knee servo holds **0.02 N·m**,
against 0.58 N·m in the standing pose. The rigid stop carries the robot for free.
**Why it matters:** on the robot the limit is a software clamp; the servo carries every gram. A
policy that learns to "lean on the stop" will sag or overheat servos on hardware.
**Tweak:** keep the command range, put the rigid stop 5° beyond it (`hard_stop_margin_deg: 5`).
The servo now holds the crouch (0.63 N·m at 93°).

### 3. Home keyframe starts 17 mm inside the floor (APPLIED)
**What:** upstream `home` puts the base at 150 mm; the soles touch at 166.9 mm.
**Why it matters:** every episode starts with a contact-solver kick that throws the robot up;
policies learn to cope with an event that never happens on hardware.
**Tweak:** base height computed from the mesh so the soles just touch.

### 4. IMU 12 mm away from the real BNO055 (APPLIED)
**What:** the `imu` site is at (−80, 0, 50) mm; the BNO055 in the CAD is at (−87, 0, 42) mm.
**Why it matters:** orientation is unaffected, but accelerometer readings include ω×r terms
from the lever arm; the brain should see what the real chip sees.
**Tweak:** site moved to the BNO055.

### 5. Servo torque limit vs. the servo you buy (OPTIONAL, randomized in training)
**What:** the sim clips at **3.23 N·m (33 kg·cm)**. `duck/build/SHOPPING.md` buys the STS3215
7.4 V, datasheet stall **19.5 kg·cm = 1.91 N·m**. (The brief quoted ~30 kg·cm at 7.4 V; that figure
belongs to the 12 V STS3215 variants. Check the label on your servos.)
**Evidence:** standing needs 0.5 N·m, a 1 s scripted rise from the deepest crouch 0.87 N·m, a 0.3 s
rise 1.26 N·m. With 125° knees (see 6) a 0.5 s rise needs 1.91 N·m, exactly a 7.4 V servo's stall,
and a 0.3 s rise wants 2.64 N·m.
**Why it matters:** fast stand-ups and get-ups trained against 3.23 N·m will stall on 1.91 N·m servos.
**Tweak:** `force_limit_nm: 1.91` if your servos are the 19.5 kg·cm ones. Independently, `env.py`
randomizes the limit between 1.9 and 3.23 N·m each episode, so policies work on either.

### 6. Knees can't fold far enough to crouch or sleep (OPTIONAL, recommended)
**What:** knee command range ±90°, but the standing pose already uses 78°. The deepest crouch lowers
the body by only 12 mm (base 159 → 147 mm, trunk bottom 121 → 109 mm).
**Evidence:** hull self-collision scan: the thigh cover first touches the foot at ~138° knee flexion;
nothing touches up to 126°. With the knee range opened to 125° the body sits 46 mm lower (base 113 mm,
trunk bottom 77 mm), holding 0.73 N·m (38% of 7.4 V stall). See `renders/crouch125_*.png`.
**Why it matters:** the sleep / deep-crouch / sit poses you want don't exist with the upstream range,
and a get-up from lying is much easier when the feet can be tucked under the body.
**Tweak:** `joint_ranges_deg: {left_knee: [-90, 125], right_knee: [-90, 125]}`. **Check on the real
leg first:** servo unpowered, fold the knee by hand, note the angle where parts touch, use that
minus 10°. Also confirm the servo's zero/offset lets it reach that angle.

### 7. Mass is CAD, not printed (OPTIONAL)
**What:** inertials come from Onshape; real prints at 15% infill weigh less than solid CAD parts,
and Bolt adds parts (foot switches, maybe a bigger battery).
**Why it matters:** mass and CoM height set what the servos can do and how the robot falls. LESSONS.md:
"weigh what you'll actually build".
**Tweak:** weigh each sub-assembly during the build and fill `masses_kg` / `payloads`. Training
randomizes mass ±10% per body anyway, but that only covers random error, not a fixed offset.

### 8. Top-heavy head on a long neck (design note, no sim change)
**What:** the head is 0.41 kg, 19% of the robot, with its CoM at 363 mm (robot CoM 205 mm), and is a
200 × 197 mm PLA shell. In every fall test the head or the trunk lands first.
**Why it matters:** falls hit the head shell and the neck servos hard; the head also dominates
balance. Fall/get-up training is where real hardware breaks.
**Proposed hardware tweak:** TPU bumpers on the head's front and back edges and the trunk's back
corners; keep the head light (no extra electronics in the head). In sim, head pitch/neck motion is
already part of the action, so the brain can use the head as a counterweight.

### 9. CoM sits towards the heels (design note)
**What:** CoM is 31 mm from the heel edge but 61 mm from the toe edge.
**Why it matters:** backward pushes and fast stops tip it over sooner than forward ones.
**Tweak (optional, for training):** sample more backward pushes; a home pose with ~2–3° more hip
pitch would centre the CoM. Not changed because upstream walking policies are tuned to this home.

### 10. Gear backlash not modelled (OPTIONAL)
**What:** the default upstream model has rigid gears; upstream ships a ±0.5° backlash variant.
**Why it matters:** cheap servo gears have a degree of play (LESSONS.md §1).
**Tweak:** `backlash: true` adds a ±0.5° free joint after each leg servo. Use it for the last stage of
walking training (it slows simulation). Note upstream's backlash model also bumps kp to 17.11; we
keep 13.37 unless your own BAM run says otherwise.

### 11. Things checked and fine
- **Joint ranges** for hips (yaw ±30°, roll ±25°, pitch −70…30° mirrored), ankle ±90°, neck
  −20…65°, head pitch ±45°, yaw ±160°, roll ±30° are reasonable for walking, looking around and
  sitting. Head can reach the trunk at neck 65° or head pitch 45°; that is now a real contact (1).
- **Self-collision** (hulls) in all crouch poses up to 126° knee: clear. Pairs that are bolted
  together (hip bracket/trunk, head mount/head) always touch and are correctly ignored.
- **Foot contact:** TPU sole 103 × 41 mm; the floor's friction (0.6, it has priority) is what the
  sim uses. Training randomizes 0.4–1.0 to cover tile, wood and rugs.
- **Solver:** 1 iteration gives the same stand-hold and lying results as 100; kept for speed.
- **Antennas** are visual only and pass through the floor when lying. They are thin wires that bend;
  fine to leave out.

## What you need to decide

1. Measure knee clearance on the real leg, then enable the 125° (or measured) knee range (6).
2. Read the label on your servos: 19.5 kg·cm → set `force_limit_nm: 1.91` (5).
3. Weigh sub-assemblies while building and enter them (7).
4. Print TPU bumpers for head and trunk before fall/get-up training on hardware (8).
