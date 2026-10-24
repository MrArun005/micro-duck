/* duck.js: the micro duck's body, balance and behaviours.
 *
 * No graphics in here, so the same code runs in the browser and in the Node
 * tests. Units: metres, seconds, kilograms, radians.
 *
 * World: +y up, ground at y = 0. Body frame: +x forward (beak), +y up, +z to
 * the duck's right. Orientation is yaw, then pitch (positive = nose up), then
 * roll (positive = top tips right): Ry * Rz * Rx.
 *
 * How it stays up (the same ideas real small bipeds are built on):
 *  - Balance: the body is a point mass on telescoping legs (the linear
 *    inverted pendulum). All the feet can do is move the centre of pressure
 *    (ZMP) around inside the soles, and the body accelerates away from it:
 *    a = ω²(c − p), with ω = √(g/h).
 *  - Capture point: ξ = c + v/ω is where a foot has to land to stop. While ξ
 *    is inside the feet the ankles can hold it; once it leaves, the duck has
 *    to step, and the swing foot is re-aimed at the predicted ξ every tick
 *    while it is in the air. If ξ runs further than a leg can reach, it falls.
 *  - Falling and getting up are a full rigid-body simulation with contact
 *    springs and friction; getting up has to work with limited motor torque.
 */
(function (root) {
  'use strict';

  const G = 9.81;
  const TAU = Math.PI * 2;
  const CANCEL = { cancelled: true };

  // About the size of a 40 cm hobby biped (Open Duck Mini class), 2 kg.
  const DUCK = {
    mass: 2.0,
    trunk: [0.17, 0.11, 0.13],            // body ellipsoid: length, height, width
    hip: [-0.005, -0.035, 0.045],         // right hip in the body frame (left mirrors z)
    thigh: 0.085, shin: 0.085,
    ankleH: 0.026,                        // ankle height above the ground when the foot is flat
    sole: [0.085, 0.012, 0.05],           // length, thickness, width
    soleFwd: 0.018,                       // sole centre sits this far ahead of the ankle
    neckBase: [0.05, 0.03, 0], neck: 0.09,
    headOffset: [0.03, 0.03, 0],          // head centre from the top of the neck, in the head frame
    head: [0.10, 0.075, 0.12],
    standH: 0.205, crouchH: 0.135, sitH: 0.088,
    stepWidth: 0.045,                     // half the distance between the feet
    maxStepFwd: 0.15, minStepSide: 0.055, maxStepSide: 0.17,
    walk: { T: 0.25, vmax: 0.22, clear: 0.025, bob: 0.005 },
    run: { T: 0.18, vmax: 0.45, clear: 0.045, bob: -0.014 },
    maxTurn: 1.6,
    getUpTorque: 4.0,                     // N·m the legs and neck can use to roll the body upright
    neckRest: -0.35,
  };

  // ---------- small vector and quaternion helpers ----------
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const len = a => Math.sqrt(dot(a, a));
  const norm = a => { const l = len(a); return l > 1e-9 ? scale(a, 1 / l) : [0, 0, 0]; };
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const wrapAngle = a => a - TAU * Math.round(a / TAU);
  const ease = s => { s = clamp(s, 0, 1); return s * s * (3 - 2 * s); };

  const qMul = (a, b) => [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
  const qAxis = (axis, ang) => { const s = Math.sin(ang / 2); return [Math.cos(ang / 2), axis[0] * s, axis[1] * s, axis[2] * s]; };
  const qX = a => qAxis([1, 0, 0], a), qY = a => qAxis([0, 1, 0], a), qZ = a => qAxis([0, 0, 1], a);
  const qNorm = q => { const l = Math.hypot(q[0], q[1], q[2], q[3]); return [q[0] / l, q[1] / l, q[2] / l, q[3] / l]; };
  function qRot(q, v) {
    const u = [q[1], q[2], q[3]], w = q[0];
    const t = scale(cross(u, v), 2);
    return add(add(v, scale(t, w)), cross(u, t));
  }
  const qConj = q => [q[0], -q[1], -q[2], -q[3]];
  const qEuler = (yaw, pitch, roll) => qMul(qMul(qY(yaw), qZ(pitch)), qX(roll));
  function eulerOf(q) {
    const x = qRot(q, [1, 0, 0]), y = qRot(q, [0, 1, 0]), z = qRot(q, [0, 0, 1]);
    return { yaw: Math.atan2(-x[2], x[0]), pitch: Math.asin(clamp(x[1], -1, 1)), roll: Math.atan2(-z[1], y[1]) };
  }

  // Deterministic random numbers so the tests are repeatable.
  function rng(seed) {
    return () => {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  // A position-controlled joint: a stiff spring-damper with a speed limit,
  // which is roughly how a serial bus servo behaves.
  class Servo {
    constructor(q, kp = 160, vmax = 7, zeta = 0.85) {
      this.q = q; this.v = 0; this.target = q;
      this.kp = kp; this.kd = 2 * zeta * Math.sqrt(kp); this.vmax = vmax;
    }
    step(dt) {
      const a = this.kp * (this.target - this.q) - this.kd * this.v;
      this.v = clamp(this.v + a * dt, -this.vmax, this.vmax);
      this.q += this.v * dt;
    }
  }

  // Two-link leg: knee position for a hip and an ankle, knee bending towards `fwd`.
  function solveKnee(hip, ankle, fwd) {
    const l1 = DUCK.thigh, l2 = DUCK.shin;
    const d0 = sub(ankle, hip);
    const d = clamp(len(d0), 0.01, l1 + l2 - 1e-4);
    const u = norm(d0);
    const n = norm(sub(fwd, scale(u, dot(fwd, u))));
    const a = (l1 * l1 - l2 * l2 + d * d) / (2 * d);
    const b = Math.sqrt(Math.max(0, l1 * l1 - a * a));
    return add(add(hip, scale(u, a)), scale(n, b));
  }

  class DuckSim {
    constructor(seed = 7) {
      this.random = rng(seed);
      this.viewer = null;           // world point the duck likes to look at (the camera)
      this.autoSleep = true;
      this.log = [];
      this.reset();
    }

    reset() {
      this.t = 0;
      this.token = 0;
      this.waiters = [];
      this.mode = 'balance';
      this.c = [0, 0]; this.cv = [0, 0];
      this.h = DUCK.sitH; this.hv = 0;
      this.yaw = 0; this.yawRate = 0;
      this.pitch = 0; this.pitchV = 0;
      this.roll = 0; this.rollV = 0;
      this.acc = [0, 0];
      this.vd = 0;
      this.cmd = { v: 0, turn: 0, gait: 'walk' };
      this.stepping = null;
      this.lastSwing = 1;
      this.steps = 0;
      this.falls = 0;
      this.feet = [-1, 1].map(side => ({ side, p: [DUCK.hip[0], DUCK.ankleH, side * DUCK.stepWidth], yaw: 0, contact: true, lift: 0 }));
      this.joints = {
        neck: new Servo(-1.15, 60, 3),
        headPitch: new Servo(0.5, 140, 6),
        headYaw: new Servo(0, 140, 7),
        headRoll: new Servo(0.15, 140, 6),
        antL: new Servo(-0.9, 220, 14, 0.35),
        antR: new Servo(-0.9, 220, 14, 0.35),
        lid: new Servo(1, 900, 30, 1),
        glow: new Servo(0.12, 30, 3, 1),
      };
      this.pose = {
        h: DUCK.sitH, pitch: 0.04, roll: 0, shift: [0, 0],
        neck: -1.15, headPitch: 0.5, headYaw: 0, headRoll: 0.15,
        antL: -0.9, antR: -0.9, lid: 1, glow: 0.12,
      };
      this.gaze = { mode: 'body', yaw: 0, pitch: 0, point: null };
      this.autoGaze = false;
      this.nextSaccade = 1;
      this.nextBlink = 2;
      this.blinkUntil = -1;
      this.idleSince = 0;
      this.activity = 'sleeping';
      this.tb = null;
      this.pending = null;
      this.debug = { xi: [0, 0], p: [0, 0], box: [], com: [0, 0] };
    }

    say(text) { this.log.push({ t: this.t, text }); if (this.log.length > 50) this.log.shift(); }

    // ---------- time and behaviours ----------

    // Run a behaviour; starting a new one cancels whatever was running.
    run(name, fn) {
      this.token++;
      const tok = this.token;
      for (const w of this.waiters) w.rej(CANCEL);
      this.waiters = [];
      this.activity = name;
      const p = fn(tok);
      return Promise.resolve(p).then(r => r, e => { if (e !== CANCEL) throw e; return 'cancelled'; });
    }

    // Resolves true when cond() holds, false after `timeout` seconds of sim time.
    until(cond, tok, timeout = Infinity) {
      return new Promise((res, rej) => {
        if (tok !== this.token) return rej(CANCEL);
        this.waiters.push({ cond, deadline: this.t + timeout, res, rej, tok });
      });
    }
    wait(s, tok) { const at = this.t + s; return this.until(() => this.t >= at, tok); }

    tickWaiters() {
      if (!this.waiters.length) return;
      const keep = [];
      for (const w of this.waiters) {
        if (w.tok !== this.token) w.rej(CANCEL);
        else if (w.cond()) w.res(true);
        else if (this.t >= w.deadline) w.res(false);
        else keep.push(w);
      }
      this.waiters = keep;
    }

    get isUp() { return this.mode === 'balance' && this.h > DUCK.standH - 0.03; }
    get canStep() { return this.mode === 'balance' && this.h > DUCK.crouchH - 0.025; }
    get gaitOn() { return Math.abs(this.cmd.v) > 0.005 || Math.abs(this.vd) > 0.01 || Math.abs(this.cmd.turn) > 0.01 || Math.abs(this.yawRate) > 0.05; }

    // What the duck is doing, for the screen.
    get state() {
      if (this.mode === 'tumble') return this.activity;
      if (this.activity === 'idle') {
        if (this.gaitOn) return this.cmd.gait === 'run' ? 'running' : 'walking';
        if (this.stepping) return 'catching its balance';
        return 'standing';
      }
      return this.activity;
    }

    // ---------- simulation ----------

    update(dt) {
      if (!(dt > 0)) return;
      const n = Math.max(1, Math.ceil(dt / (1 / 240) - 1e-6));
      for (let i = 0; i < n; i++) this.step(dt / n);
    }

    step(dt) {
      this.t += dt;
      this.tickWaiters();
      this.life(dt);
      if (this.mode === 'balance') this.stepBalance(dt);
      else for (let i = 0; i < 4 && this.mode === 'tumble'; i++) this.stepTumble(dt / 4);
      this.aimHead();
      for (const k in this.joints) this.joints[k].step(dt);
    }

    heading() { return [Math.cos(this.yaw), -Math.sin(this.yaw)]; }
    sideDir() { return [Math.sin(this.yaw), Math.cos(this.yaw)]; }
    soleCentre(f) {
      return [f.p[0] + Math.cos(f.yaw) * DUCK.soleFwd, f.p[2] - Math.sin(f.yaw) * DUCK.soleFwd];
    }

    // Support area of the feet on the ground, as a box in the heading frame.
    supportBox(feet) {
      const hd = this.heading(), sd = this.sideDir();
      let f0 = Infinity, f1 = -Infinity, l0 = Infinity, l1 = -Infinity;
      for (const f of feet) {
        const s = this.soleCentre(f);
        const pf = s[0] * hd[0] + s[1] * hd[1], pl = s[0] * sd[0] + s[1] * sd[1];
        f0 = Math.min(f0, pf - DUCK.sole[0] / 2); f1 = Math.max(f1, pf + DUCK.sole[0] / 2);
        l0 = Math.min(l0, pl - DUCK.sole[2] / 2); l1 = Math.max(l1, pl + DUCK.sole[2] / 2);
      }
      return { hd, sd, f0, f1, l0, l1 };
    }
    toBox(b, p) { return [p[0] * b.hd[0] + p[1] * b.hd[1], p[0] * b.sd[0] + p[1] * b.sd[1]]; }
    fromBox(b, f, l) { return [b.hd[0] * f + b.sd[0] * l, b.hd[1] * f + b.sd[1] * l]; }
    clampBox(b, p) { const [f, l] = this.toBox(b, p); return this.fromBox(b, clamp(f, b.f0, b.f1), clamp(l, b.l0, b.l1)); }
    outside(b, p) {
      const [f, l] = this.toBox(b, p);
      return Math.hypot(Math.max(0, b.f0 - f, f - b.f1), Math.max(0, b.l0 - l, l - b.l1));
    }
    boxCorners(b) {
      return [[b.f0, b.l0], [b.f1, b.l0], [b.f1, b.l1], [b.f0, b.l1]].map(([f, l]) => this.fromBox(b, f, l));
    }

    stepBalance(dt) {
      const gait = DUCK[this.cmd.gait];
      const hc = Math.max(0.06, this.h + 0.015);         // centre of mass sits a little above the trunk centre (the head)
      const w = Math.sqrt(G / hc);

      // Smooth the commands like a real controller would.
      const vt = this.canStep ? clamp(this.cmd.v, -gait.vmax * 0.5, gait.vmax) : 0;
      this.vd += clamp(vt - this.vd, -0.9 * dt, 0.9 * dt);
      // Turning at speed needs sideways grip and sideways steps, so the faster
      // it goes the gentler it turns (about 0.3 m/s² of sideways acceleration).
      const turnMax = Math.min(DUCK.maxTurn, 0.3 / Math.max(0.05, Math.abs(this.vd)));
      const tt = this.canStep ? clamp(this.cmd.turn, -turnMax, turnMax) : 0;
      this.yawRate += clamp(tt - this.yawRate, -5 * dt, 5 * dt);

      const xi = [this.c[0] + this.cv[0] / w, this.c[1] + this.cv[1] / w];
      let p, box;

      if (!this.stepping) {
        box = this.supportBox(this.feet);
        if (this.gaitOn && this.canStep) {
          // Shift the weight over the stance foot first, then lift the other one.
          const st = this.feet[this.lastSwing], sc = this.soleCentre(st);
          const inb = -st.side * 0.012;
          const target = [sc[0] + box.sd[0] * inb, sc[1] + box.sd[1] * inb];
          p = this.clampBox(box, [xi[0] + 2 * (xi[0] - target[0]), xi[1] + 2 * (xi[1] - target[1])]);
          if (Math.hypot(xi[0] - target[0], xi[1] - target[1]) < 0.015) this.startStep(1 - this.lastSwing);
        } else {
          const mid = [(this.soleCentre(this.feet[0])[0] + this.soleCentre(this.feet[1])[0]) / 2,
                       (this.soleCentre(this.feet[0])[1] + this.soleCentre(this.feet[1])[1]) / 2];
          const target = [mid[0] + box.hd[0] * this.pose.shift[0] + box.sd[0] * this.pose.shift[1],
                          mid[1] + box.hd[1] * this.pose.shift[0] + box.sd[1] * this.pose.shift[1]];
          p = this.clampBox(box, [xi[0] + 1.5 * (xi[0] - target[0]), xi[1] + 1.5 * (xi[1] - target[1])]);
          const out = this.outside(box, xi);
          if (out > 0.008 && this.canStep) this.startStep(this.chooseRecoveryFoot(xi, box));
          else if (out > 0.03) { this.fall(); return; }
        }
      }

      if (this.stepping) {
        const st = this.feet[1 - this.stepping.swing];
        box = this.supportBox([st]);
        // The ankle shifts the pressure inside the sole: a little while walking,
        // as much as it can when catching a push.
        const k = this.stepping.recover ? 0.85 : 0.3;
        const cen = this.soleCentre(st);
        p = this.clampBox(box, [cen[0] + k * (xi[0] - cen[0]), cen[1] + k * (xi[1] - cen[1])]);
        if (Math.hypot(xi[0] - p[0], xi[1] - p[1]) > 0.26) { this.fall(); return; }
        this.updateSwing(dt, xi, cen, w * (1 - k));
      }

      // The pendulum: the body accelerates away from the centre of pressure.
      const a = [w * w * (this.c[0] - p[0]), w * w * (this.c[1] - p[1])];
      this.acc = a;
      this.cv[0] += a[0] * dt; this.cv[1] += a[1] * dt;
      this.c[0] += this.cv[0] * dt; this.c[1] += this.cv[1] * dt;
      if (this.stepping || this.gaitOn) this.yaw = wrapAngle(this.yaw + this.yawRate * dt);

      // Body lean follows the acceleration, height follows the pose, with a bob per step.
      const hd = this.heading(), sd = this.sideDir();
      const af = a[0] * hd[0] + a[1] * hd[1], al = a[0] * sd[0] + a[1] * sd[1];
      const vf = this.cv[0] * hd[0] + this.cv[1] * hd[1];
      const pitchT = this.pose.pitch - clamp(0.03 * af + 0.35 * vf, -0.45, 0.45);
      const rollT = this.pose.roll + clamp(0.025 * al, -0.3, 0.3) + 0.12 * this.yawRate * Math.max(0, vf);
      this.pitchV += (150 * (pitchT - this.pitch) - 20 * this.pitchV) * dt;
      this.rollV += (150 * (rollT - this.roll) - 20 * this.rollV) * dt;
      this.pitch += this.pitchV * dt; this.roll += this.rollV * dt;

      let hT = this.pose.h;
      if (this.activity === 'sleeping') hT += 0.0035 * Math.sin(this.t * 1.7);     // breathing
      if (this.stepping) {
        const s = this.stepping.t / this.stepping.T, b = this.stepping.recover ? 0 : DUCK[this.stepping.gait].bob;
        hT += b * (Math.sin(Math.PI * s) - 0.5);
      }
      const ha = clamp(70 * (hT - this.h) - 15 * this.hv, -G, 2.5 * G);
      this.hv += ha * dt; this.h += this.hv * dt;

      this.debug = { xi, p, com: this.c.slice(), box: this.boxCorners(box) };
    }

    chooseRecoveryFoot(xi, box) {
      const mid = [(this.feet[0].p[0] + this.feet[1].p[0]) / 2, (this.feet[0].p[2] + this.feet[1].p[2]) / 2];
      const [f, l] = this.toBox(box, [xi[0] - mid[0], xi[1] - mid[1]]);
      if (Math.abs(l) > 0.4 * Math.abs(f)) return l > 0 ? 1 : 0;   // step out with the foot on that side
      const d = this.feet.map(ft => { const s = this.soleCentre(ft); return Math.hypot(s[0] - xi[0], s[1] - xi[1]); });
      return d[0] > d[1] ? 0 : 1;
    }

    // Step timing follows the speed the body actually has, not just the button:
    // switching to slow walking steps while still moving at running speed would
    // need steps longer than the legs, so the duck keeps short running steps
    // until it has slowed down.
    stepGait() {
      if (this.cmd.gait === 'run') return 'run';
      // vd is the smoothed speed the controller is tracking; the body's own
      // speed swings up and down within every step, so it is not used here.
      const speed = Math.abs(this.vd);
      if (speed > DUCK.walk.vmax * 1.02) return 'run';
      if (this.stepping && this.stepping.gait === 'run' && speed > DUCK.walk.vmax * 0.85) return 'run';
      return 'walk';
    }

    startStep(swing) {
      const f = this.feet[swing];
      const gait = this.stepGait();
      const recover = !(this.gaitOn && this.canStep);
      this.stepping = { swing, gait, recover, t: 0, T: recover ? 0.2 : DUCK[gait].T, from: this.soleCentre(f), fromYaw: f.yaw, target: this.soleCentre(f) };
      f.contact = false;
    }

    // Aim the swing foot at where the capture point will be when it lands.
    updateSwing(dt, xi, stance, w) {
      const S = this.stepping, f = this.feet[S.swing];
      S.t += dt;
      const tau = Math.max(0, S.T - S.t);
      const grow = Math.exp(w * tau), E = Math.exp(w * S.T);
      const xiEnd = [stance[0] + (xi[0] - stance[0]) * grow, stance[1] + (xi[1] - stance[1]) * grow];
      const hd = this.heading(), sd = this.sideDir();
      const sigma = f.side;
      // Walking lands the foot a little behind the predicted capture point so
      // the body keeps rolling forward at speed vd; a catch step aims right at it.
      const dF = S.recover ? 0 : this.vd * S.T / Math.max(0.2, E - 1);
      const dL = 2 * DUCK.stepWidth / (E + 1);
      let tgt = [xiEnd[0] - hd[0] * dF + sd[0] * sigma * dL, xiEnd[1] - hd[1] * dF + sd[1] * sigma * dL];
      // Legs only reach so far, and must not cross.
      let rf = (tgt[0] - stance[0]) * hd[0] + (tgt[1] - stance[1]) * hd[1];
      let rl = (tgt[0] - stance[0]) * sd[0] + (tgt[1] - stance[1]) * sd[1];
      // Like a real controller, slow down rather than overreach.
      if (!S.recover && Math.abs(rf) > 0.85 * DUCK.maxStepFwd) this.vd -= Math.sign(rf) * 1.5 * dt;
      rf = clamp(rf, -DUCK.maxStepFwd, DUCK.maxStepFwd);
      rl = sigma > 0 ? clamp(rl, DUCK.minStepSide, DUCK.maxStepSide) : clamp(rl, -DUCK.maxStepSide, -DUCK.minStepSide);
      // Don't re-aim much in the last moments; a real foot can't teleport.
      const blend = S.t / S.T > 0.8 ? 0.15 : 1;
      const want = [stance[0] + hd[0] * rf + sd[0] * rl, stance[1] + hd[1] * rf + sd[1] * rl];
      S.target = [S.target[0] + (want[0] - S.target[0]) * blend, S.target[1] + (want[1] - S.target[1]) * blend];
      tgt = S.target;

      const s = clamp(S.t / S.T, 0, 1), e = ease(s);
      const sole = [S.from[0] + (tgt[0] - S.from[0]) * e, S.from[1] + (tgt[1] - S.from[1]) * e];
      f.yaw = S.fromYaw + wrapAngle(this.yaw - S.fromYaw) * e;
      f.lift = Math.sin(Math.PI * s);
      f.p = [sole[0] - Math.cos(f.yaw) * DUCK.soleFwd, DUCK.ankleH + DUCK[S.gait].clear * f.lift, sole[1] + Math.sin(f.yaw) * DUCK.soleFwd];

      if (S.t >= S.T) {
        f.p[1] = DUCK.ankleH; f.lift = 0; f.contact = true; f.yaw = this.yaw;
        this.stepping = null;
        this.lastSwing = S.swing;
        this.steps++;
        // Keep the gait going, but after a catch step shift the weight first
        // (the no-step branch above does that) instead of launching straight on.
        if (this.gaitOn && this.canStep && !S.recover) { this.startStep(1 - S.swing); return; }
        // Recovering from a push: keep stepping until the feet can hold the capture point.
        const box = this.supportBox(this.feet);
        // (w here is the slowed-down rate under the ankle; the capture point uses the real one)
        const w0 = Math.sqrt(G / Math.max(0.06, this.h + 0.015));
        const xiNow = [this.c[0] + this.cv[0] / w0, this.c[1] + this.cv[1] / w0];
        if (this.outside(box, xiNow) > 0.004 || Math.hypot(this.cv[0], this.cv[1]) > 0.25) this.startStep(1 - S.swing);
      }
    }

    // ---------- falling: rigid body with contacts ----------

    fall(spin = 2.5) {
      if (this.mode === 'tumble') return;
      this.falls++;
      this.stepping = null;
      const q = qEuler(this.yaw, this.pitch, this.roll);
      const pos = [this.c[0], this.h, this.c[1]];
      const vdir = norm([this.cv[0], 0, this.cv[1]]);
      const w = add(scale(cross([0, 1, 0], vdir), spin), qRot(qY(this.yaw), [this.rollV, 0, this.pitchV]));
      const toBody = v => qRot(qConj(q), sub(v, pos));
      this.tb = {
        pos, q, w,
        vel: [this.cv[0], this.hv, this.cv[1]],
        ankles: this.feet.map(f => toBody(f.p)),
        still: 0, settled: false, motor: false,
      };
      this.mode = 'tumble';
      this.vd = 0; this.cmd.v = 0; this.cmd.turn = 0; this.yawRate = 0;
      this.afterFall();
    }

    tumblePoints() {
      const pts = [];
      const [lx, ly, lz] = DUCK.trunk.map(v => v / 2);
      for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) pts.push([sx * lx * 0.7, sy * ly * 0.7, sz * lz * 0.7]);
      pts.push([lx, 0, 0], [-lx, 0, 0], [0, ly, 0], [0, -ly, 0], [0, 0, lz], [0, 0, -lz]);
      const hk = this.headLocal();
      const [hx, hy, hz] = DUCK.head.map(v => v / 2);
      for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) pts.push(add(hk.centre, qRot(hk.q, [sx * hx, sy * hy, sz * hz])));
      for (const a of this.tb.ankles) {
        const down = DUCK.ankleH - DUCK.sole[1];
        pts.push(add(a, [DUCK.soleFwd + DUCK.sole[0] / 2, -down, 0]), add(a, [DUCK.soleFwd - DUCK.sole[0] / 2, -down, 0]));
      }
      return pts;
    }

    stepTumble(dt) {
      const B = this.tb, m = DUCK.mass;
      const F = [0, -m * G, 0];
      let T = [0, 0, 0];
      const K = 3000, C = 25, CT = 30, MU = 0.8;
      for (const pb of this.tumblePoints()) {
        const rel = qRot(B.q, pb);
        const y = B.pos[1] + rel[1];
        if (y >= 0) continue;
        const vp = add(B.vel, cross(B.w, rel));
        const fn = Math.max(0, -K * y - C * vp[1]);
        let ft = [-CT * vp[0], 0, -CT * vp[2]];
        const fl = len(ft);
        if (fl > MU * fn) ft = scale(ft, MU * fn / fl);
        const f = [ft[0], fn, ft[2]];
        F[0] += f[0]; F[1] += f[1]; F[2] += f[2];
        T = add(T, cross(rel, f));
      }

      const up = qRot(B.q, [0, 1, 0]);
      const upness = up[1];
      if (B.motor) {
        // Legs and neck push against the ground to roll the body upright.
        let err = cross(up, [0, 1, 0]);
        const s = len(err), ang = Math.atan2(s, upness);
        let axis;
        if (upness < -0.5) { const xa = qRot(B.q, [1, 0, 0]); axis = scale(xa, dot(err, xa) >= 0 ? 1 : -1); }
        else axis = s > 1e-4 ? scale(err, 1 / s) : [0, 0, 0];
        let tm = sub(scale(axis, 7 * ang), scale([B.w[0], 0, B.w[2]], 0.5));
        tm[1] -= 0.15 * B.w[1];
        const tl = len(tm);
        if (tl > DUCK.getUpTorque) tm = scale(tm, DUCK.getUpTorque / tl);
        T = add(T, tm);
        // Fold the feet under the hips while getting up.
        const k = Math.min(1, 2.2 * dt);
        B.ankles = B.ankles.map((a, i) => {
          const side = i === 0 ? -1 : 1;
          const tuck = [DUCK.hip[0], -(DUCK.sitH - DUCK.ankleH) + 0.006, side * DUCK.stepWidth];
          return add(a, scale(sub(tuck, a), k));
        });
      }

      B.vel = add(B.vel, scale(F, dt / m));
      B.pos = add(B.pos, scale(B.vel, dt));
      // Inertia of a 2 kg duck-shaped box, rotated into the world.
      const Ib = [0.0095, 0.0082, 0.012];
      const tb = qRot(qConj(B.q), T);
      const wb = qRot(qConj(B.q), B.w);
      const Iw = [Ib[0] * wb[0], Ib[1] * wb[1], Ib[2] * wb[2]];
      const gyro = cross(wb, Iw);
      const dwb = [(tb[0] - gyro[0]) / Ib[0], (tb[1] - gyro[1]) / Ib[1], (tb[2] - gyro[2]) / Ib[2]];
      B.w = add(B.w, scale(qRot(B.q, dwb), dt));
      B.w = scale(B.w, 1 - 0.4 * dt);
      const dq = qMul([0, B.w[0], B.w[1], B.w[2]], B.q);
      B.q = qNorm([B.q[0] + 0.5 * dt * dq[0], B.q[1] + 0.5 * dt * dq[1], B.q[2] + 0.5 * dt * dq[2], B.q[3] + 0.5 * dt * dq[3]]);

      const slow = len(B.vel) < 0.06 && len(B.w) < 0.4;
      B.still = slow ? B.still + dt : 0;
      if (B.still > 0.35) B.settled = true;

      if (B.motor && upness > 0.97 && len(B.w) < 1.5 && B.pos[1] > 0.06) this.landOnFeet();
      this.debug = { xi: [B.pos[0], B.pos[2]], p: [B.pos[0], B.pos[2]], com: [B.pos[0], B.pos[2]], box: [] };
    }

    // Upright again: hand the body back to the balance controller.
    landOnFeet() {
      const B = this.tb;
      const e = eulerOf(B.q);
      this.mode = 'balance';
      this.yaw = e.yaw; this.pitch = e.pitch; this.roll = e.roll;
      this.pitchV = 0; this.rollV = 0;
      this.c = [B.pos[0], B.pos[2]]; this.cv = [B.vel[0] * 0.5, B.vel[2] * 0.5];
      this.h = B.pos[1]; this.hv = Math.max(0, B.vel[1]);
      this.pose.h = this.h; this.pose.pitch = 0;
      this.feet = B.ankles.map((a, i) => {
        const wpt = add(B.pos, qRot(B.q, a));
        return { side: i === 0 ? -1 : 1, p: [wpt[0], DUCK.ankleH, wpt[2]], yaw: e.yaw, contact: true, lift: 0 };
      });
      this.stepping = null;
      this.tb = null;
    }

    // ---------- head, eyes, antennas ----------

    headLocal() {
      const J = this.joints;
      const nt = add(DUCK.neckBase, qRot(qZ(J.neck.q), [0, DUCK.neck, 0]));
      const q = qMul(qMul(qMul(qZ(J.neck.q), qY(J.headYaw.q)), qZ(J.headPitch.q)), qX(J.headRoll.q));
      return { neckTop: nt, q, centre: add(nt, qRot(q, DUCK.headOffset)) };
    }

    trunkQuat() { return this.mode === 'tumble' ? this.tb.q : qEuler(this.yaw, this.pitch, this.roll); }
    trunkPos() { return this.mode === 'tumble' ? this.tb.pos : [this.c[0], this.h, this.c[1]]; }

    aimHead() {
      const J = this.joints, P = this.pose;
      J.neck.target = P.neck;
      J.antL.target = P.antL; J.antR.target = P.antR;
      J.glow.target = P.glow;
      J.lid.target = this.t < this.blinkUntil ? 1 : P.lid;
      if (this.gaze.mode === 'body' || this.mode === 'tumble') {
        J.headPitch.target = P.headPitch; J.headYaw.target = P.headYaw; J.headRoll.target = P.headRoll;
        return;
      }
      // Keep the head steady in the world, like a bird does, and look at the target.
      let gy = this.gaze.yaw, gp = this.gaze.pitch;
      if (this.gaze.point) {
        const hc = add(this.trunkPos(), qRot(this.trunkQuat(), this.headLocal().centre));
        const d = sub(this.gaze.point, hc);
        gy = Math.atan2(-d[2], d[0]);
        gp = Math.atan2(d[1], Math.hypot(d[0], d[2]));
      }
      J.headYaw.target = clamp(wrapAngle(gy - this.yaw), -1.4, 1.4);
      J.headPitch.target = clamp(gp - this.pitch - J.neck.q, -0.9, 1.1);
      J.headRoll.target = clamp(-this.roll + P.headRoll, -0.6, 0.6);
    }

    // Small things a living creature does by itself: blink, glance, get sleepy.
    life() {
      const busy = this.activity !== 'idle';
      const awake = this.joints.glow.target > 0.5 && this.mode === 'balance';
      if (awake && this.t > this.nextBlink) {
        this.blinkUntil = this.t + 0.11;
        this.nextBlink = this.t + 1.8 + this.random() * 3.5 + (this.random() < 0.2 ? -1.5 : 0);
        if (this.random() < 0.15) this.nextBlink = this.t + 0.25;     // double blink
      }
      if (this.autoGaze && this.t > this.nextSaccade) {
        const r = this.random();
        if (this.gaitOn) {
          this.gaze = { mode: 'look', yaw: this.yaw + 0.5 * this.yawRate + (this.random() - 0.5) * 0.4, pitch: -0.15, point: null };
          this.nextSaccade = this.t + 0.8 + this.random();
        } else if (r < 0.3 && this.viewer) {
          this.gaze = { mode: 'look', yaw: 0, pitch: 0, point: this.viewer.slice() };
          this.nextSaccade = this.t + 2 + this.random() * 2.5;
        } else {
          this.gaze = { mode: 'look', yaw: this.yaw + (this.random() - 0.5) * 1.8, pitch: -0.25 + this.random() * 0.5, point: null };
          this.nextSaccade = this.t + 1 + this.random() * 2.5;
        }
      }
      if (busy || this.gaitOn || this.stepping) this.idleSince = this.t;
      if (this.autoSleep && !busy && this.isUp && this.t - this.idleSince > 45) this.goToSleep();
    }

    wiggleAntennas(tok, times = 3, amp = 0.5) {
      const P = this.pose;
      return (async () => {
        for (let i = 0; i < times; i++) {
          P.antL = amp; P.antR = -amp; await this.wait(0.09, tok);
          P.antL = -amp; P.antR = amp; await this.wait(0.09, tok);
        }
        P.antL = 0; P.antR = 0;
      })();
    }

    // ---------- behaviours ----------

    goToSleep() {
      if (this.mode !== 'balance') return false;
      if (this.activity === 'sleeping') return true;
      this.run('getting sleepy', async tok => {
        const P = this.pose, W = s => this.wait(s, tok);
        this.cmd.v = 0; this.cmd.turn = 0;
        await this.until(() => !this.stepping && Math.abs(this.vd) < 0.02, tok, 4);
        this.autoGaze = false;
        this.gaze = { mode: 'look', yaw: this.yaw, pitch: 0.05, point: null };
        P.lid = 0.45; P.antL = -0.2; P.antR = -0.2; await W(0.8);
        // a big yawn
        P.neck = -0.05; this.gaze.pitch = 0.55; P.lid = 0.85; P.antL = 0.3; P.antR = 0.3; await W(1.2);
        this.gaze.pitch = -0.05; P.neck = DUCK.neckRest; P.lid = 0.55; P.antL = -0.4; P.antR = -0.4; await W(0.6);
        this.activity = 'sitting down';
        P.pitch = 0.04; P.h = DUCK.crouchH; await W(0.5);
        P.h = DUCK.sitH; await W(0.9);
        // nodding off
        this.activity = 'nodding off';
        this.gaze = { mode: 'body' };
        for (let i = 0; i < 2; i++) {
          P.neck = -0.95; P.headPitch = 0.35; P.headYaw = 0; P.headRoll = 0.1; P.lid = 0.9; await W(1.0 + 0.4 * i);
          P.neck = -0.45; P.headPitch = 0.3; P.lid = 0.4; P.antL = 0.2; P.antR = 0.2; await W(0.35);
          P.antL = -0.5; P.antR = -0.5; await W(0.4);
        }
        P.neck = -1.15; P.headPitch = 0.5; P.headRoll = 0.15; P.lid = 1; P.glow = 0.12; P.antL = -0.9; P.antR = -0.9;
        this.say('Zzz…');
        this.activity = 'sleeping';
      });
      return true;
    }

    wake() {
      if (this.mode !== 'balance') return false;
      if (this.activity === 'idle' && this.isUp) return true;
      this.run('waking up', async tok => {
        const P = this.pose, W = s => this.wait(s, tok);
        if (this.joints.glow.target < 0.5) {
          this.gaze = { mode: 'body' };
          P.headRoll = 0;
          for (const [lid, s] of [[0.55, 0.3], [1, 0.25], [0.3, 0.35], [0.9, 0.15], [0.1, 0.4], [0.5, 0.12], [0, 0.2]]) {
            P.lid = lid; P.glow = 1 - 0.6 * lid; await W(s);
          }
          P.glow = 1;
          P.neck = -0.25; P.headPitch = 0.15; P.antL = 0; P.antR = 0; await W(0.5);
          // stretch
          this.activity = 'stretching';
          P.neck = 0.2; P.headPitch = 0.45; P.lid = 0.6; P.antL = 0.6; P.antR = 0.6; await W(1.0);
          P.neck = DUCK.neckRest; P.headPitch = 0.3; P.lid = 0; await W(0.4);
          // shake it off
          for (let i = 0; i < 4; i++) { P.headRoll = i % 2 ? -0.35 : 0.35; await W(0.11); }
          P.headRoll = 0;
          await this.wiggleAntennas(tok, 2);
          // where am I?
          this.activity = 'looking around';
          this.gaze = { mode: 'look', yaw: this.yaw + 0.85, pitch: 0.1, point: null }; await W(0.8);
          this.gaze = { mode: 'look', yaw: this.yaw - 0.85, pitch: 0.05, point: null }; await W(0.8);
          this.gaze = this.viewer ? { mode: 'look', point: this.viewer.slice() } : { mode: 'look', yaw: this.yaw, pitch: 0, point: null };
          await W(0.6);
          this.say('Good morning!');
        }
        await this.standUpSeq(tok);
      });
      return true;
    }

    async standUpSeq(tok) {
      const P = this.pose, W = s => this.wait(s, tok);
      if (this.h < DUCK.standH - 0.02) {
        this.activity = 'standing up';
        this.autoGaze = false;
        this.gaze = { mode: 'look', yaw: this.yaw, pitch: -0.15, point: null };
        P.lid = 0; P.glow = 1;
        P.neck = -0.75; P.pitch = -0.38; P.h = DUCK.crouchH; await W(0.55);
        P.h = DUCK.standH; P.pitch = -0.12; await W(0.45);
        P.pitch = 0; P.neck = DUCK.neckRest; await W(0.35);
        await this.wiggleAntennas(tok, 2, 0.35);
      }
      P.h = DUCK.standH; P.pitch = 0; P.neck = DUCK.neckRest; P.lid = 0; P.glow = 1; P.antL = 0; P.antR = 0; P.headRoll = 0;
      P.shift = [0, 0];
      this.autoGaze = true;
      this.nextSaccade = this.t + 0.3;
      this.idleSince = this.t;
      this.activity = 'idle';
      if (this.pending) {
        this.cmd.gait = this.pending.gait; this.cmd.v = this.pending.v; this.cmd.turn = 0;
        this.pending = null;
      }
    }

    standUp() {
      if (this.mode !== 'balance') return false;
      if (this.joints.glow.target < 0.5) return this.wake();
      this.run('standing up', tok => this.standUpSeq(tok));
      return true;
    }

    sit() {
      if (this.mode !== 'balance') return false;
      this.run('sitting down', async tok => {
        const P = this.pose, W = s => this.wait(s, tok);
        this.cmd.v = 0; this.cmd.turn = 0;
        await this.until(() => !this.stepping && Math.abs(this.vd) < 0.02, tok, 4);
        P.pitch = 0.05; P.h = DUCK.crouchH; await W(0.45);
        P.h = DUCK.sitH; P.glow = 1; await W(0.8);
        this.activity = 'sitting';
        this.autoGaze = true;
      });
      return true;
    }

    // Keyboard and buttons: drive with a forward speed and a turn rate.
    drive(v, turn, gait = 'walk') {
      if (this.mode !== 'balance') return false;
      if (!this.isUp || this.activity !== 'idle') {
        if (this.activity === 'idle' || this.activity === 'sitting' || this.activity === 'sleeping') {
          this.wake();
        }
        return false;
      }
      this.cmd.gait = gait; this.cmd.v = v; this.cmd.turn = turn;
      this.idleSince = this.t;
      return true;
    }

    move(gait, v) {
      if (this.mode !== 'balance') return false;
      const go = { gait, v: v ?? DUCK[gait].vmax * (gait === 'run' ? 1 : 0.85) };
      if (this.joints.glow.target < 0.5) { this.pending = go; return this.wake(); }
      this.run(gait === 'run' ? 'getting ready to run' : 'getting ready to walk', async tok => {
        this.pending = go;
        await this.standUpSeq(tok);
      });
      return true;
    }

    stop() { this.cmd.v = 0; this.cmd.turn = 0; return true; }

    // Turn 90°. Standing still it turns on the spot; when it is already moving
    // it eases off to a slow walk for the turn (never stopping dead, which at
    // running speed would need a step longer than its legs), then picks its
    // old speed and gait back up.
    turn(dir) {
      if (this.mode !== 'balance' || !this.isUp) return this.standUp();
      const was = this.gaitOn && this.cmd.v > 0.02 ? { v: this.cmd.v, gait: this.cmd.gait } : null;
      this.run('turning', async tok => {
        const start = this.yaw, goal = Math.PI / 2;
        // Keep the step rhythm it already has; only the speed comes down.
        this.cmd.gait = was ? was.gait : 'walk';
        this.cmd.v = was ? Math.min(was.v, 0.6 * DUCK.walk.vmax) : 0;
        // Ease into the turn while still fast; full turn rate once slowed down.
        this.cmd.turn = dir * 0.5;
        this.autoGaze = false;
        this.gaze = { mode: 'look', yaw: start + dir * goal, pitch: -0.05, point: null };
        await this.until(() => this.vd < DUCK.walk.vmax * 0.8, tok, 3);
        this.cmd.turn = dir * 1.3;
        await this.until(() => Math.abs(wrapAngle(this.yaw - start)) > goal - 0.12, tok, 6);
        this.cmd.turn = 0;
        if (was) {
          this.cmd.gait = was.gait; this.cmd.v = was.v;
          await this.until(() => Math.abs(this.yawRate) < 0.05, tok, 2);
        } else {
          await this.until(() => Math.abs(this.yawRate) < 0.05 && !this.stepping, tok, 3);
        }
        this.autoGaze = true;
        this.activity = 'idle';
      });
      return true;
    }

    lookAround() {
      if (this.mode !== 'balance') return false;
      if (this.joints.glow.target < 0.5) return this.wake();
      const wasActivity = this.activity;
      this.run('looking around', async tok => {
        const W = s => this.wait(s, tok), y = this.yaw;
        this.autoGaze = false;
        const spots = [[1.1, 0.3], [0.6, -0.4], [-0.3, 0.6], [-1.1, 0.0], [-0.5, -0.45]];
        for (const [dy, dp] of spots) {
          this.gaze = { mode: 'look', yaw: y + dy, pitch: dp, point: null };
          this.pose.headRoll = (this.random() - 0.5) * 0.5;
          await W(0.55 + this.random() * 0.4);
        }
        if (this.viewer) { this.gaze = { mode: 'look', point: this.viewer.slice() }; this.pose.headRoll = 0.3; await this.wiggleAntennas(tok, 2); await W(0.6); }
        this.pose.headRoll = 0;
        this.autoGaze = true;
        this.activity = wasActivity === 'sitting' ? 'sitting' : 'idle';
      });
      return true;
    }

    dance() {
      if (this.mode !== 'balance') return false;
      this.run('dancing', async tok => {
        const P = this.pose, W = s => this.wait(s, tok);
        if (!this.isUp || this.joints.glow.target < 0.5) {
          if (this.joints.glow.target < 0.5) { this.wake(); return; }
          await this.standUpSeq(tok);
          this.activity = 'dancing';
        }
        this.cmd.v = 0; this.cmd.turn = 0;
        await this.until(() => !this.stepping, tok, 3);
        this.gaze = this.viewer ? { mode: 'look', point: this.viewer.slice() } : { mode: 'look', yaw: this.yaw, pitch: 0, point: null };
        this.autoGaze = false;
        const beat = 0.3;
        for (let i = 0; i < 8; i++) {
          const s = i % 2 ? 1 : -1;
          P.shift = [0, 0.018 * s]; P.headRoll = -0.3 * s; P.antL = 0.6 * s; P.antR = -0.6 * s;
          P.h = DUCK.standH - 0.025; await W(beat / 2);
          P.h = DUCK.standH; await W(beat / 2);
        }
        P.shift = [0, 0]; P.headRoll = 0;
        // a little bow
        P.pitch = -0.3; P.neck = -0.9; P.h = DUCK.crouchH + 0.03; await W(0.6);
        await this.standUpSeq(tok);
      });
      return true;
    }

    // Shove the duck: `dir` is a horizontal world direction, `speed` in m/s.
    push(dir, speed) {
      const d = norm([dir[0], 0, dir[1]]);
      if (this.mode === 'tumble') {
        this.tb.vel = add(this.tb.vel, scale(d, speed));
        this.tb.w = add(this.tb.w, scale(cross([0, 1, 0], d), speed * 6));
        this.tb.settled = false; this.tb.still = 0;
        return;
      }
      this.cv[0] += d[0] * speed; this.cv[1] += d[2] * speed;
      const local = qRot(qY(-this.yaw), d);
      this.pitchV -= local[0] * speed * 6;
      this.rollV += local[2] * speed * 6;
      this.blinkUntil = this.t + 0.06;
      if (this.activity === 'idle') {
        this.gaze = { mode: 'look', yaw: Math.atan2(d[2], -d[0]), pitch: 0, point: null };
        this.nextSaccade = this.t + 1.2;
        this.pose.antL = 0.7; this.pose.antR = 0.7;
        setTimeoutSim(this, 0.4, () => { if (this.activity === 'idle') { this.pose.antL = 0; this.pose.antR = 0; } });
      }
      if (this.activity === 'sleeping' && speed < 0.5) this.wake();
    }

    afterFall() {
      this.run('falling', async tok => {
        const P = this.pose, W = s => this.wait(s, tok);
        this.autoGaze = false;
        this.gaze = { mode: 'body' };
        P.headPitch = 0.4; P.headYaw = 0; P.headRoll = 0; P.neck = -0.2;
        P.lid = 0; P.glow = 1; P.antL = 0.9; P.antR = 0.9;
        this.say('Whoa!');
        await this.until(() => this.tb && this.tb.settled, tok, 6);
        this.activity = 'fallen';
        P.antL = -0.6; P.antR = 0.4;
        // dizzy
        for (let i = 0; i < 6; i++) { P.headYaw = Math.sin(i * 1.3) * 0.5; P.headRoll = Math.cos(i * 1.3) * 0.3; P.lid = i % 2 ? 0.5 : 0.2; await W(0.22); }
        P.headYaw = 0; P.headRoll = 0; P.lid = 0;
        for (let attempt = 0; this.mode === 'tumble' && attempt < 4; attempt++) {
          this.activity = 'getting up';
          P.neck = -0.6; P.headPitch = 0.6; P.antL = 0; P.antR = 0;
          this.tb.motor = true;
          const ok = await this.until(() => this.mode === 'balance', tok, 5);
          if (ok) break;
          this.tb.motor = false; P.lid = 0.4;
          await W(0.8);
          P.lid = 0;
        }
        if (this.mode === 'tumble') { this.activity = 'stuck'; this.say('I need a hand…'); return; }
        await this.standUpSeq(tok);
        this.activity = 'shaking it off';
        for (let i = 0; i < 4; i++) { P.headRoll = i % 2 ? -0.3 : 0.3; await W(0.1); }
        P.headRoll = 0;
        this.activity = 'idle';
        this.say('I\'m OK!');
      });
    }

    // ---------- everything the renderer needs, in world space ----------

    kinematics() {
      const q = this.trunkQuat(), pos = this.trunkPos();
      const W = v => add(pos, qRot(q, v));
      const hk = this.headLocal();
      const fwd = qRot(q, [1, 0, 0]);
      const legs = [-1, 1].map((side, i) => {
        const hip = W([DUCK.hip[0], DUCK.hip[1], side * DUCK.hip[2]]);
        let ankle, footQ, lift = 0;
        if (this.mode === 'tumble') {
          ankle = W(this.tb.ankles[i]);
          footQ = q;
        } else {
          const f = this.feet[i];
          ankle = f.p; lift = f.lift;
          footQ = qMul(qY(f.yaw), qZ(0.35 * f.lift * (this.stepping && this.stepping.swing === i ? 1 : 0)));
        }
        const knee = solveKnee(hip, ankle, fwd);
        return { hip, knee, ankle, footQ, lift };
      });
      const J = this.joints;
      return {
        pos, q,
        neckBase: W(DUCK.neckBase), neckTop: W(hk.neckTop),
        headPos: W(hk.centre), headQ: qMul(q, hk.q),
        legs,
        antL: J.antL.q, antR: J.antR.q, lid: clamp(J.lid.q, 0, 1), glow: clamp(J.glow.q, 0, 1),
        sleeping: this.activity === 'sleeping',
      };
    }
  }

  // Run fn after `s` seconds of simulated time (cancelled by nothing; used for small reactions).
  function setTimeoutSim(sim, s, fn) {
    const at = sim.t + s;
    sim.waiters.push({ cond: () => sim.t >= at, deadline: Infinity, res: fn, rej: () => {}, tok: sim.token });
  }

  const api = { DuckSim, DUCK, G, qRot, qMul, qEuler, eulerOf, solveKnee };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MicroDuck = api;
})(typeof window !== 'undefined' ? window : globalThis);
