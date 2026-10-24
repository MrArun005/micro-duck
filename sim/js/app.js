/* app.js: wires the duck simulation, the 3D view, buttons, keyboard and the
 * live readout together. Every element is optional, so the same file drives
 * sim/index.html and smaller embeds of it. */
(() => {
  'use strict';
  const { DuckSim } = window.MicroDuck;
  const $ = id => document.getElementById(id);

  const duck = new DuckSim((Date.now() % 997) + 1);
  window.duck = duck;               // handy from the console
  let view = null;
  try {
    if (!window.THREE) throw new Error('three.js did not load');
    view = new MicroDuckView.DuckView($('scene'));
    const mq = matchMedia('(prefers-color-scheme: dark)');
    mq.addEventListener && mq.addEventListener('change', () => view.applyTheme());
    new MutationObserver(() => view.applyTheme()).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class', 'style'] });
    view.onPick = dir => duck.push(dir, 0.35);
  } catch (e) {
    console.warn(e);
    if ($('nogl')) $('nogl').style.display = 'grid';
  }

  // ---------- buttons ----------
  const fwd = () => [Math.cos(duck.yaw), -Math.sin(duck.yaw)];
  const awayFromCamera = () => {
    if (!view) return fwd();
    const c = view.cameraPos(), p = duck.trunkPos();
    return [p[0] - c[0], p[2] - c[2]];
  };
  const actions = {
    wake: () => duck.wake(),
    sleep: () => duck.goToSleep(),
    stand: () => duck.standUp(),
    sit: () => duck.sit(),
    walk: () => duck.move('walk'),
    run: () => duck.move('run'),
    left: () => duck.turn(1),
    right: () => duck.turn(-1),
    stop: () => duck.stop(),
    look: () => duck.lookAround(),
    dance: () => duck.dance(),
    push: () => duck.push(awayFromCamera(), 0.3),
    shove: () => duck.push(awayFromCamera(), 1.5),
  };
  document.querySelectorAll('[data-act]').forEach(b => b.addEventListener('click', () => {
    const f = actions[b.dataset.act];
    if (f) f();
    b.blur();
  }));

  const dbgBtn = $('debugBtn');
  if (dbgBtn) dbgBtn.addEventListener('click', () => {
    const on = dbgBtn.getAttribute('aria-pressed') !== 'true';
    dbgBtn.setAttribute('aria-pressed', String(on));
    if (view) view.showDebug = on;
    if ($('legend')) $('legend').hidden = !on;
  });

  // ---------- keyboard: WASD / arrows drive, Shift runs, Space sleeps or wakes ----------
  const keys = new Set();
  let driving = false;
  const isTyping = e => /input|textarea|select/i.test(e.target.tagName || '');
  addEventListener('keydown', e => {
    if (isTyping(e) || e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === ' ') {
      e.preventDefault();
      if (!e.repeat) (duck.activity === 'sleeping' || duck.joints.glow.target < 0.5 ? duck.wake() : duck.goToSleep());
      return;
    }
    if (['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'shift'].includes(k)) {
      if (k.startsWith('arrow')) e.preventDefault();
      keys.add(k);
    }
  });
  addEventListener('keyup', e => keys.delete(e.key.toLowerCase()));
  addEventListener('blur', () => keys.clear());

  function readKeys() {
    const f = (keys.has('w') || keys.has('arrowup') ? 1 : 0) - (keys.has('s') || keys.has('arrowdown') ? 1 : 0);
    const t = (keys.has('a') || keys.has('arrowleft') ? 1 : 0) - (keys.has('d') || keys.has('arrowright') ? 1 : 0);
    const moving = f !== 0 || t !== 0;
    if (moving) {
      const run = keys.has('shift') && f > 0;
      const v = f > 0 ? (run ? 0.45 : 0.19) : f < 0 ? -0.08 : 0;
      if (duck.drive(v, t * 1.3, run ? 'run' : 'walk')) driving = true;
    } else if (driving) {
      duck.drive(0, 0, duck.cmd.gait);
      driving = false;
    }
  }

  // ---------- readout ----------
  const stateEl = $('state'), speedEl = $('speed'), heightEl = $('height'), stepsEl = $('steps'), fallsEl = $('falls'), sayEl = $('say');
  let lastSay = 0, sayTimer = 0;
  function readout() {
    if (stateEl) stateEl.textContent = duck.state;
    const v = duck.mode === 'tumble' ? Math.hypot(duck.tb.vel[0], duck.tb.vel[2]) : Math.hypot(duck.cv[0], duck.cv[1]);
    if (speedEl) speedEl.textContent = v.toFixed(2) + ' m/s';
    if (heightEl) heightEl.textContent = (duck.trunkPos()[1] * 100).toFixed(1) + ' cm';
    if (stepsEl) stepsEl.textContent = duck.steps;
    if (fallsEl) fallsEl.textContent = duck.falls;
    if (sayEl && duck.log.length && duck.log[duck.log.length - 1] !== lastSay) {
      lastSay = duck.log[duck.log.length - 1];
      sayEl.textContent = lastSay.text;
      sayEl.classList.add('show');
      clearTimeout(sayTimer);
      sayTimer = setTimeout(() => sayEl.classList.remove('show'), 2200);
    }
  }

  // ---------- main loop ----------
  let last = performance.now(), acc = 0;
  function frame(now) {
    // rAF timestamps can be a little older than performance.now(), so never go backwards.
    const realDt = Math.min(0.1, Math.max(0, (now - last) / 1000));
    last = now;
    readKeys();
    duck.update(Math.min(1 / 20, realDt));
    if (view) {
      view.render(duck, realDt);
      duck.viewer = view.cameraPos();
    }
    acc += realDt;
    if (acc > 0.1) { acc = 0; readout(); }
    requestAnimationFrame(frame);
  }
  readout();
  requestAnimationFrame(frame);
})();
