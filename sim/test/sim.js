// Drive the duck simulation with a fixed time step. Behaviours are async
// (they await sim-time waits), so yield to the event loop every few steps to
// let their promises resolve, the same way pet/test/run.js does.
const { DuckSim, DUCK } = require('../js/duck.js');
const DT = 1 / 240;
const tick = () => new Promise(r => setImmediate(r));

async function advance(duck, seconds, stopWhen) {
  const n = Math.round(seconds / DT);
  for (let k = 0; k < n; k++) {
    duck.step(DT);
    if (k % 4 === 0) {
      await tick();
      if (stopWhen && stopWhen(duck)) return true;
    }
  }
  return stopWhen ? !!stopWhen(duck) : true;
}

// A duck that is awake, standing and idle (it starts asleep).
async function standingDuck(seed = 4) {
  const duck = new DuckSim(seed);
  duck.autoSleep = false;
  duck.wake();
  const ok = await advance(duck, 20, d => d.activity === 'idle' && d.isUp);
  if (!ok) throw new Error(`duck never stood up (state: ${duck.state})`);
  await advance(duck, 0.5);
  return duck;
}

const speed = d => Math.hypot(d.cv[0], d.cv[1]);
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const angle = a => a - 2 * Math.PI * Math.round(a / (2 * Math.PI));

module.exports = { DuckSim, DUCK, advance, standingDuck, speed, dist, angle };
