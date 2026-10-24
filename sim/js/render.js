/* render.js: draws the micro duck with three.js. It only reads the
 * simulation (duck.kinematics() and duck.debug); it never changes it, so the
 * physics in duck.js stays usable on its own (tests, training, a real robot). */
(function (root) {
  'use strict';
  const THREE = root.THREE, M = root.MicroDuck;
  const { DUCK } = M;
  const UP = new THREE.Vector3(0, 1, 0);
  const v3 = a => new THREE.Vector3(a[0], a[1], a[2]);
  const q4 = q => new THREE.Quaternion(q[1], q[2], q[3], q[0]);   // sim quats are [w, x, y, z]
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  // Colours are written in sRGB; the renderer works in linear light.
  const col = hex => new THREE.Color(hex).convertSRGBToLinear();

  function cssColor(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    try { return new THREE.Color(v || fallback); } catch (e) { return new THREE.Color(fallback); }
  }

  class DuckView {
    constructor(canvas) {
      this.canvas = canvas;
      this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
      this.renderer.setPixelRatio(Math.min(2, root.devicePixelRatio || 1));
      this.renderer.shadowMap.enabled = true;
      this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      this.renderer.outputEncoding = THREE.sRGBEncoding;
      this.scene = new THREE.Scene();
      this.camera = new THREE.PerspectiveCamera(38, 1, 0.02, 30);
      this.home = { az: 0.95, el: 0.32, dist: 0.95 };
      this.orbit = Object.assign({ target: new THREE.Vector3(0, 0.14, 0) }, this.home);
      this.showDebug = false;
      this.onPick = null;          // called with a world direction when the duck is clicked
      this.raycaster = new THREE.Raycaster();

      this.buildWorld();
      this.buildDuck();
      this.buildDebug();
      this.buildZzz();
      this.applyTheme();
      this.bindOrbit();
      new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
      this.resize();
    }

    buildWorld() {
      const s = this.scene;
      this.hemi = new THREE.HemisphereLight(0xffffff, 0x8a8577, 0.7);
      s.add(this.hemi);
      const sun = this.sun = new THREE.DirectionalLight(0xffffff, 0.75);
      sun.castShadow = true;
      sun.shadow.mapSize.set(1024, 1024);
      sun.shadow.radius = 4;
      sun.shadow.bias = -0.0004;
      sun.shadow.normalBias = 0.002;      // no shadow acne on the curved shells
      Object.assign(sun.shadow.camera, { left: -0.6, right: 0.6, top: 0.6, bottom: -0.6, near: 0.2, far: 5 });
      s.add(sun, sun.target);

      this.floorMat = new THREE.MeshStandardMaterial({ roughness: 0.95 });
      const floor = this.floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), this.floorMat);
      floor.rotation.x = -Math.PI / 2;
      floor.receiveShadow = true;
      s.add(floor);
      // 10 cm grid, 1 m lines a little stronger.
      this.grid = new THREE.GridHelper(20, 200);
      this.grid.material.transparent = true; this.grid.material.opacity = 0.3; this.grid.material.vertexColors = false;
      this.grid.position.y = 0.0005;
      this.grid2 = new THREE.GridHelper(20, 20);
      this.grid2.material.transparent = true; this.grid2.material.opacity = 0.6; this.grid2.material.vertexColors = false;
      this.grid2.position.y = 0.0008;
      s.add(this.grid, this.grid2);
    }

    buildDuck() {
      const cast = m => { m.castShadow = true; m.receiveShadow = true; return m; };
      this.shellMat = new THREE.MeshStandardMaterial({ color: col(0xf3ecdc), roughness: 0.45, metalness: 0.05 });
      this.darkMat = new THREE.MeshStandardMaterial({ color: col(0x2b2f33), roughness: 0.55, metalness: 0.2 });
      this.visorMat = new THREE.MeshStandardMaterial({ color: col(0x101416), roughness: 0.15, metalness: 0.4 });
      this.orangeMat = new THREE.MeshStandardMaterial({ color: col(0xf07a12), roughness: 0.5 });
      this.jointMat = new THREE.MeshStandardMaterial({ color: col(0x4a5056), roughness: 0.4, metalness: 0.5 });
      this.eyeMat = new THREE.MeshBasicMaterial({ color: 0x5fdcff });
      const shell = this.shellMat, dark = this.darkMat;

      // Body: a rounded egg with a darker hip plate and a perky tail.
      const body = this.body = new THREE.Group();
      const [bl, bh, bw] = DUCK.trunk;
      const egg = cast(new THREE.Mesh(new THREE.SphereGeometry(0.5, 40, 28), shell));
      egg.scale.set(bl, bh, bw);
      body.add(egg);
      const belly = cast(new THREE.Mesh(new THREE.SphereGeometry(0.5, 32, 20), dark));
      belly.scale.set(bl * 0.62, bh * 0.5, bw * 0.92);
      belly.position.set(-0.005, -0.03, 0);
      body.add(belly);
      // a short, flat, upturned tail
      const tail = cast(new THREE.Mesh(new THREE.ConeGeometry(0.016, 0.034, 20), shell));
      tail.position.set(-bl / 2 + 0.004, 0.024, 0);
      tail.rotation.z = Math.PI / 2 + 0.75;
      tail.scale.set(1, 1, 0.55);
      body.add(tail);
      // a small status light on the back
      this.backLight = new THREE.Mesh(new THREE.SphereGeometry(0.006, 12, 8), this.eyeMat);
      this.backLight.position.set(-0.035, bh / 2 - 0.002, 0);
      body.add(this.backLight);
      [-1, 1].forEach(s => {
        const hipCap = cast(new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.012, 20), this.jointMat));
        hipCap.rotation.x = Math.PI / 2;
        hipCap.position.set(DUCK.hip[0], DUCK.hip[1], s * (DUCK.hip[2] + 0.006));
        body.add(hipCap);
      });
      this.scene.add(body);

      // Neck: stretched between its base and top each frame.
      this.neck = cast(new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.019, 1, 18), dark));
      this.scene.add(this.neck);

      // Head: rounded cream shell with a glossy dark face visor.
      const head = this.head = new THREE.Group();
      const [hl, hh, hw] = DUCK.head;
      const dome = cast(new THREE.Mesh(new THREE.SphereGeometry(0.5, 36, 24), shell));
      dome.scale.set(hl, hh * 1.05, hw);
      head.add(dome);
      // A slice of a slightly bigger ellipsoid, centred on +x (phi = π faces forward).
      const visor = cast(new THREE.Mesh(new THREE.SphereGeometry(0.5, 48, 32, Math.PI - 1.25, 2.5, Math.PI * 0.2, Math.PI * 0.55), this.visorMat));
      visor.scale.set(hl * 1.04, hh * 1.02, hw * 1.04);
      visor.receiveShadow = false; dome.receiveShadow = false;   // self-shadow speckles on the curved head
      head.add(visor);
      // eyes: glowing discs with eyelids made by squashing them
      this.eyes = [-1, 1].map(s => {
        const g = new THREE.Group();
        const ey = 0.006, ez = s * 0.026;
        const ex = hl / 2 * 1.04 * Math.sqrt(1 - (ey / (hh / 2 * 1.02)) ** 2 - (ez / (hw / 2 * 1.04)) ** 2) + 0.0008;
        g.position.set(ex, ey, ez);
        g.rotation.y = Math.PI / 2 - s * 0.42;
        const eye = new THREE.Mesh(new THREE.CircleGeometry(0.0105, 24), this.eyeMat);
        g.add(eye);
        head.add(g);
        return eye;
      });
      // bill: a small flattened orange lozenge under the visor
      const bill = cast(new THREE.Mesh(new THREE.SphereGeometry(0.5, 24, 12), this.orangeMat));
      bill.scale.set(0.036, 0.011, 0.034);
      bill.position.set(hl / 2 - 0.002, -0.025, 0);
      head.add(bill);
      // antennas on pivots at the top of the head
      this.antennas = [-1, 1].map(s => {
        const pivot = new THREE.Group();
        pivot.rotation.x = s * 0.25;            // splay outwards
        const stalk = cast(new THREE.Mesh(new THREE.CylinderGeometry(0.0022, 0.003, 0.055, 8), dark));
        stalk.position.y = 0.0275;
        const tip = new THREE.Mesh(new THREE.SphereGeometry(0.0065, 14, 10), this.orangeMat);
        tip.position.y = 0.057; tip.castShadow = true;
        pivot.add(stalk, tip);
        const holder = new THREE.Group();       // swings forward/back (rotation.z)
        holder.position.set(-0.008, hh / 2 - 0.006, s * 0.03);
        holder.add(pivot);
        head.add(holder);
        return holder;
      });
      this.scene.add(head);

      // Legs: thigh and shin stretched between joints, flat orange webbed feet.
      const footShape = new THREE.Shape();
      const L = DUCK.sole[0], W = DUCK.sole[2];
      footShape.moveTo(-0.025, -0.012);
      footShape.quadraticCurveTo(-0.032, 0, -0.025, 0.012);
      footShape.lineTo(L - 0.03, W / 2 + 0.002);
      footShape.quadraticCurveTo(L - 0.02, W / 2 - 0.006, L - 0.024, W / 6);
      footShape.quadraticCurveTo(L - 0.016, 0, L - 0.024, -W / 6);
      footShape.quadraticCurveTo(L - 0.02, -W / 2 + 0.006, L - 0.03, -W / 2 - 0.002);
      footShape.lineTo(-0.025, -0.012);
      const footGeom = new THREE.ExtrudeGeometry(footShape, { depth: DUCK.sole[1] - 0.004, bevelEnabled: true, bevelThickness: 0.002, bevelSize: 0.0015, bevelSegments: 2 });
      footGeom.rotateX(Math.PI / 2);            // lie flat; shape y becomes world z
      footGeom.translate(-0.012, DUCK.sole[1], 0);
      const cyl = (r1, r2, mat) => cast(new THREE.Mesh(new THREE.CylinderGeometry(r1, r2, 1, 16), mat));
      const ball = (r, mat) => cast(new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), mat));
      this.legs = [0, 1].map(() => {
        const leg = {
          thigh: cyl(0.012, 0.0105, shell), shin: cyl(0.011, 0.009, dark),
          knee: ball(0.014, this.jointMat), ankle: ball(0.01, this.jointMat),
          foot: new THREE.Group(),
        };
        const foot = cast(new THREE.Mesh(footGeom, this.orangeMat));
        foot.position.y = -(DUCK.ankleH);
        leg.foot.add(foot);
        for (const k in leg) this.scene.add(leg[k]);
        return leg;
      });
      this.pickables = [egg, belly, tail, dome, visor, bill];
    }

    buildDebug() {
      const g = this.debug = new THREE.Group();
      const lineGeom = new THREE.BufferGeometry();
      lineGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(5 * 3), 3));
      this.supportLine = new THREE.Line(lineGeom, new THREE.LineBasicMaterial({ color: 0x2f9e44 }));
      this.supportLine.frustumCulled = false;
      const fillGeom = new THREE.BufferGeometry();
      fillGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6 * 3), 3));
      this.supportFill = new THREE.Mesh(fillGeom, new THREE.MeshBasicMaterial({ color: 0x2f9e44, transparent: true, opacity: 0.25, side: THREE.DoubleSide, depthWrite: false }));
      this.supportFill.frustumCulled = false;
      const dot = (c, r) => new THREE.Mesh(new THREE.CylinderGeometry(r, r, 0.002, 24), new THREE.MeshBasicMaterial({ color: c }));
      this.xiDot = dot(0xf08a1c, 0.009);
      this.zmpDot = dot(0x1c7ed6, 0.007);
      this.comDot = dot(0x868e96, 0.005);
      const stemGeom = new THREE.BufferGeometry();
      stemGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
      this.comStem = new THREE.Line(stemGeom, new THREE.LineDashedMaterial({ color: 0x868e96, dashSize: 0.01, gapSize: 0.008 }));
      this.comStem.frustumCulled = false;
      g.add(this.supportFill, this.supportLine, this.xiDot, this.zmpDot, this.comDot, this.comStem);
      g.visible = false;
      this.scene.add(g);
    }

    buildZzz() {
      const c = document.createElement('canvas');
      c.width = c.height = 64;
      const x = c.getContext('2d');
      x.font = '700 52px system-ui, sans-serif';
      x.textAlign = 'center'; x.textBaseline = 'middle';
      x.fillStyle = '#ffffff';
      x.fillText('z', 32, 34);
      const tex = new THREE.CanvasTexture(c);
      this.zzz = [0, 1, 2].map(() => {
        const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
        s.visible = false;
        this.scene.add(s);
        return s;
      });
    }

    applyTheme() {
      const bg = cssColor('--scene-bg', '#e9ece6');
      this.scene.background = bg;
      this.scene.fog = new THREE.Fog(bg, 1.6, 6);
      // The sun and sky light the floor above its own colour; tone it down so it
      // ends up close to the token, and the grid stays visible.
      this.floorMat.color = cssColor('--scene-floor', '#f4f5f1').multiplyScalar(0.72);
      const grid = cssColor('--scene-grid', '#9aa49c');
      this.grid.material.color = grid; this.grid2.material.color = grid;
      const z = cssColor('--muted', '#5d6a61');
      this.zzz.forEach(s => s.material.color = z);
      this.eyeBase = cssColor('--eye', '#5fdcff');
    }

    bindOrbit() {
      const c = this.canvas, pts = new Map();
      let pinch = 0, moved = 0, downAt = null;
      c.addEventListener('pointerdown', e => {
        c.setPointerCapture(e.pointerId);
        pts.set(e.pointerId, [e.clientX, e.clientY]);
        if (pts.size === 1) { moved = 0; downAt = [e.clientX, e.clientY]; }
      });
      const end = e => {
        if (pts.size === 1 && downAt && moved < 6 && e.type === 'pointerup') this.pick(e.clientX, e.clientY);
        pts.delete(e.pointerId); pinch = 0;
      };
      c.addEventListener('pointerup', end);
      c.addEventListener('pointercancel', end);
      c.addEventListener('pointermove', e => {
        if (!pts.has(e.pointerId)) return;
        const [px, py] = pts.get(e.pointerId);
        pts.set(e.pointerId, [e.clientX, e.clientY]);
        if (downAt) moved = Math.max(moved, Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]));
        if (pts.size === 1) {
          if (moved < 6) return;
          this.orbit.az -= (e.clientX - px) * 0.008;
          this.orbit.el = clamp(this.orbit.el + (e.clientY - py) * 0.006, 0.04, 1.4);
        } else if (pts.size === 2) {
          moved = 99;
          const [a, b] = [...pts.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
          if (pinch) this.zoom(pinch / d);
          pinch = d;
        }
      });
      c.addEventListener('wheel', e => { e.preventDefault(); this.zoom(Math.exp(e.deltaY * 0.001)); }, { passive: false });
      c.addEventListener('dblclick', () => Object.assign(this.orbit, this.home));
    }
    zoom(f) { this.orbit.dist = clamp(this.orbit.dist * f, 0.35, 3.5); }

    // Clicking the duck pushes it away from the camera.
    pick(x, y) {
      if (!this.onPick) return;
      const r = this.canvas.getBoundingClientRect();
      this.raycaster.setFromCamera({ x: (x - r.left) / r.width * 2 - 1, y: -((y - r.top) / r.height * 2 - 1) }, this.camera);
      const hit = this.raycaster.intersectObjects(this.pickables, false)[0];
      if (!hit) return;
      const d = hit.point.clone().sub(this.camera.position);
      this.onPick([d.x, d.z], hit.point);
    }

    resize() {
      const p = this.canvas.parentElement, w = p.clientWidth, h = p.clientHeight;
      if (!w || !h) return;
      this.renderer.setSize(w, h, false);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    }

    cameraPos() { const p = this.camera.position; return [p.x, p.y, p.z]; }

    render(duck, realDt) {
      const K = duck.kinematics(), o = this.orbit;
      // The camera follows the body, gently, and ignores small bobs.
      const k = 1 - Math.exp(-realDt * 4);
      // Lead a little in the direction of travel so a running duck stays in frame.
      const vel = duck.mode === 'tumble' ? [duck.tb.vel[0], duck.tb.vel[2]] : duck.cv;
      o.target.x += (K.pos[0] + 0.25 * vel[0] - o.target.x) * k;
      o.target.z += (K.pos[2] + 0.25 * vel[1] - o.target.z) * k;
      o.target.y += (clamp(K.pos[1], 0.06, 0.2) * 0.75 + 0.03 - o.target.y) * k;
      // Narrow (phone) views back off so the whole duck still fits.
      const dist = o.dist * Math.max(1, 1.1 / Math.max(0.3, this.camera.aspect));
      this.camera.position.set(
        o.target.x + dist * Math.cos(o.el) * Math.sin(o.az),
        o.target.y + dist * Math.sin(o.el),
        o.target.z + dist * Math.cos(o.el) * Math.cos(o.az));
      this.camera.lookAt(o.target);
      this.sun.position.set(o.target.x + 0.8, 2.2, o.target.z + 1.2);
      this.sun.target.position.copy(o.target);

      // Body and head straight from the simulation.
      this.body.position.set(...K.pos);
      this.body.quaternion.copy(q4(K.q));
      this.stretch(this.neck, K.neckBase, K.neckTop);
      this.head.position.set(...K.headPos);
      this.head.quaternion.copy(q4(K.headQ));
      this.antennas[0].rotation.z = K.antL;
      this.antennas[1].rotation.z = K.antR;

      // Eyes: the lid squashes the disc from the top; glow dims when sleepy.
      const open = Math.max(0.06, 1 - K.lid);
      this.eyes.forEach(e => { e.scale.y = open; e.position.y = 0.0105 * (open - 1) * 0.6; });
      const glow = 0.18 + 0.82 * K.glow;
      this.eyeMat.color.copy(this.eyeBase || new THREE.Color(0x5fdcff)).multiplyScalar(glow);
      this.backLight.visible = K.glow > 0.5 || Math.sin(duck.t * 2) > 0.6;

      // Legs.
      K.legs.forEach((L, i) => {
        const m = this.legs[i];
        this.stretch(m.thigh, L.hip, L.knee);
        this.stretch(m.shin, L.knee, L.ankle);
        m.knee.position.set(...L.knee);
        m.ankle.position.set(...L.ankle);
        m.foot.position.set(...L.ankle);
        m.foot.quaternion.copy(q4(L.footQ));
      });

      // Physics overlay: support area, capture point, ZMP, centre of mass.
      this.debug.visible = this.showDebug;
      if (this.showDebug) this.drawDebug(duck, K);

      // Floating z's while asleep.
      this.zzz.forEach((s, i) => {
        s.visible = K.sleeping;
        if (!K.sleeping) return;
        const ph = (duck.t * 0.35 + i / 3) % 1;
        s.position.set(K.headPos[0] - 0.01 + ph * 0.04, K.headPos[1] + 0.05 + ph * 0.14, K.headPos[2] + Math.sin(ph * 6 + i) * 0.015);
        const sz = 0.025 + ph * 0.03;
        s.scale.set(sz, sz, 1);
        s.material.opacity = Math.sin(Math.PI * ph);
      });

      this.renderer.render(this.scene, this.camera);
    }

    drawDebug(duck, K) {
      const D = duck.debug, y = 0.002;
      const box = D.box || [];
      const show = box.length === 4;
      this.supportLine.visible = this.supportFill.visible = show;
      if (show) {
        const a = this.supportLine.geometry.attributes.position;
        [0, 1, 2, 3, 0].forEach((j, i) => a.setXYZ(i, box[j][0], y, box[j][1]));
        a.needsUpdate = true;
        const f = this.supportFill.geometry.attributes.position;
        [0, 1, 2, 0, 2, 3].forEach((j, i) => f.setXYZ(i, box[j][0], y * 0.8, box[j][1]));
        f.needsUpdate = true;
      }
      this.xiDot.position.set(D.xi[0], y + 0.002, D.xi[1]);
      this.zmpDot.position.set(D.p[0], y + 0.003, D.p[1]);
      this.comDot.position.set(D.com[0], y + 0.001, D.com[1]);
      const s = this.comStem.geometry.attributes.position;
      s.setXYZ(0, K.pos[0], K.pos[1], K.pos[2]); s.setXYZ(1, D.com[0], y, D.com[1]);
      s.needsUpdate = true;
      this.comStem.computeLineDistances();
    }

    stretch(mesh, a, b) {
      const A = v3(a), B = v3(b), d = B.clone().sub(A), L = d.length();
      mesh.position.copy(A).addScaledVector(d, 0.5);
      mesh.scale.set(1, Math.max(L, 1e-3), 1);
      if (L > 1e-6) mesh.quaternion.setFromUnitVectors(UP, d.normalize());
    }
  }

  root.MicroDuckView = { DuckView };
})(self);
