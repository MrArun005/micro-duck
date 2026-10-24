const test = require('node:test');
const assert = require('node:assert/strict');
const { DuckSim, DUCK, advance, standingDuck, dist, angle } = require('./sim.js');

test('stands still for 5 s without falling', async () => {
  const d = await standingDuck();
  const c0 = d.c.slice();
  await advance(d, 5);
  assert.equal(d.falls, 0);
  assert.equal(d.state, 'standing');
  assert.ok(d.h > DUCK.standH - 0.01, `height ${d.h}`);
  assert.ok(dist(d.c, c0) < 0.02, 'did not wander off');
});

test('a small push is caught with a few steps, no fall', async () => {
  const d = await standingDuck();
  const steps = d.steps;
  d.push([1, 0], 0.3);
  await advance(d, 3);
  assert.equal(d.falls, 0);
  assert.ok(d.steps - steps >= 1, 'it stepped to catch itself');
  assert.ok(d.steps - steps <= 6, `took ${d.steps - steps} steps`);
  assert.equal(d.state, 'standing', 'stops stepping once balanced');
});

test('a firmer push is caught and the stepping stops', async () => {
  const d = await standingDuck();
  d.push([1, 0], 0.45);
  await advance(d, 4);
  assert.equal(d.falls, 0);
  assert.equal(d.state, 'standing', `still ${d.state}`);
});

test('pushes from every side are caught', async () => {
  for (let i = 0; i < 8; i++) {
    const d = await standingDuck(10 + i);
    const a = i * Math.PI / 4;
    d.push([Math.cos(a), Math.sin(a)], 0.3);
    await advance(d, 3);
    assert.equal(d.falls, 0, `push towards ${(a * 180 / Math.PI).toFixed(0)}°`);
    assert.equal(d.state, 'standing');
  }
});

test('a big shove knocks it over, and it is standing again within 15 s', async () => {
  const d = await standingDuck();
  d.push([1, 0], 1.5);
  await advance(d, 2);
  assert.equal(d.falls, 1, 'it fell');
  assert.equal(d.mode, 'tumble');
  const up = await advance(d, 15, x => x.mode === 'balance' && x.activity === 'idle' && x.isUp);
  assert.ok(up, `still ${d.state} after 15 s`);
  await advance(d, 2);
  assert.equal(d.falls, 1, 'stays up afterwards');
});

test('walking for 3 s covers more than 0.4 m', async () => {
  const d = await standingDuck();
  d.move('walk');
  await advance(d, 3, x => x.activity === 'idle');
  const c0 = d.c.slice();
  await advance(d, 3);
  assert.equal(d.falls, 0);
  assert.equal(d.state, 'walking');
  assert.ok(dist(d.c, c0) > 0.4, `walked ${dist(d.c, c0).toFixed(2)} m`);
});

test('running is faster than walking', async () => {
  const pace = async gait => {
    const d = await standingDuck();
    d.move(gait);
    await advance(d, 4);
    const c0 = d.c.slice();
    await advance(d, 3);
    assert.equal(d.falls, 0, gait);
    return dist(d.c, c0) / 3;
  };
  const walk = await pace('walk'), run = await pace('run');
  assert.ok(run > walk * 1.6, `run ${run.toFixed(2)} m/s vs walk ${walk.toFixed(2)} m/s`);
});

test('turns 90° on the spot', async () => {
  for (const dir of [1, -1]) {
    const d = await standingDuck();
    const y0 = d.yaw;
    const r = d.turn(dir);
    assert.equal(r, true);
    const done = await advance(d, 8, x => x.activity === 'idle');
    assert.ok(done, 'turn finished');
    const turned = angle(d.yaw - y0);
    assert.ok(Math.abs(turned - dir * Math.PI / 2) < 0.2, `turned ${turned.toFixed(2)} rad`);
    assert.equal(d.falls, 0);
  }
});

test('sleep, wake and stand up runs through to standing idle', async () => {
  const d = await standingDuck();
  d.goToSleep();
  assert.ok(await advance(d, 15, x => x.activity === 'sleeping'), `stuck in ${d.state}`);
  assert.ok(d.h < DUCK.crouchH, 'sits down to sleep');
  await advance(d, 2);
  assert.ok(d.joints.glow.q < 0.3, 'eyes dim');
  assert.ok(d.kinematics().sleeping);
  d.wake();
  assert.ok(await advance(d, 20, x => x.activity === 'idle'), `stuck in ${d.state}`);
  assert.equal(d.state, 'standing');
  assert.ok(d.isUp);
  assert.ok(d.joints.lid.target < 0.5, 'eyes open');
  d.sit();
  assert.ok(await advance(d, 5, x => x.activity === 'sitting'));
  d.standUp();
  assert.ok(await advance(d, 5, x => x.activity === 'idle' && x.isUp));
  assert.equal(d.falls, 0);
});

test('turning out of a full-speed run does not fall', async () => {
  for (const dir of [1, -1]) {
    for (const extra of [0, 0.07, 0.13]) {      // different points in the stride
      const d = await standingDuck();
      d.move('run');
      await advance(d, 4 + extra);
      assert.ok(d.vd > 0.4, 'at full speed');
      const y0 = d.yaw;
      d.turn(dir);
      await advance(d, 6);
      assert.equal(d.falls, 0, `dir ${dir}, +${extra}s`);
      assert.ok(Math.abs(angle(d.yaw - y0) - dir * Math.PI / 2) < 0.25, 'turned about 90°');
    }
  }
});

test('walking and turning with the keyboard controls stays up', async () => {
  const d = await standingDuck();
  const plan = [[0.19, 0, 'walk', 2], [0.19, 1.3, 'walk', 3], [0.45, 0, 'run', 3], [0.45, -1.3, 'run', 3],
                [0.19, 1.3, 'walk', 2], [0, 0, 'walk', 1.5], [-0.08, 0, 'walk', 2], [0.45, 0.6, 'run', 2], [0, 0, 'walk', 2]];
  for (const [v, turn, gait, s] of plan) {
    d.drive(v, turn, gait);
    await advance(d, s);
    assert.equal(d.falls, 0, `driving ${gait} ${v} turn ${turn}`);
  }
});

test('a long walk does not drift into a fall', async () => {
  const d = await standingDuck();
  d.move('walk');
  await advance(d, 60);
  assert.equal(d.falls, 0);
  assert.ok(Math.abs(d.c[1]) < 0.15, `sideways drift ${d.c[1].toFixed(3)} m over ${d.c[0].toFixed(1)} m`);
});

test('kinematics: leg segments keep their length', async () => {
  const d = await standingDuck();
  d.move('walk');
  await advance(d, 2);
  for (const L of d.kinematics().legs) {
    assert.ok(Math.abs(dist3(L.hip, L.knee) - DUCK.thigh) < 1e-6);
    const s = dist3(L.knee, L.ankle);
    assert.ok(s <= DUCK.shin + 1e-6 && s > DUCK.shin - 0.02, `shin ${s}`);
  }
});
const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
