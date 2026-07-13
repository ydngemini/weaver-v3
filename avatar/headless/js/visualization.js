import {
  canvas, state, key, THREE_MODULE_URL, setVisualBoot, configurePerformance,
} from './core.js';
import { CORE_QUBITS, QUANTUM_ARCHITECTURE } from './visual-data.js';
import {
  buildVisualAudit, deriveVisualSignals, effectiveDpr,
  recordRenderedFrame, recordSkippedFrame, stableUnit,
} from './visual-runtime.js';

let ctx = null;
let resizeRaf = 0;

function canUseWebGL() {
  try {
    const probe = document.createElement('canvas');
    return !!(window.WebGLRenderingContext && (probe.getContext('webgl2') || probe.getContext('webgl')));
  } catch (e) {
    return false;
  }
}
async function initReactiveField() {
  configurePerformance();
  setVisualBoot(0.14, 'Checking graphics support…');
  if (state.reducedMotion || !canUseWebGL()) {
    initFallbackField();
    return;
  }
  state.visualMode = 'loading-3d';
  setVisualBoot(0.28, 'Loading the quantum field…');
  try {
    const THREE = await import(THREE_MODULE_URL);
    setVisualBoot(0.62, 'Building the cortex manifold…');
    createThreeField(THREE);
    setVisualBoot(0.90, 'Calibrating responsive motion…');
    resize();
    startDraw();
    setVisualBoot(1, 'Reactive field ready.', true);
  } catch (e) {
    state.visualMode = 'fallback-2d';
    state.lastVisualError = e.message || String(e);
    initFallbackField();
  }
}
function initFallbackField() {
  if (!ctx) ctx = canvas.getContext('2d', { alpha: false });
  state.visualMode = state.visualMode === 'fallback-2d' ? state.visualMode : '2d';
  resize();
  startDraw();
  setVisualBoot(1, 'Efficient reactive field ready.', true);
}
function createLightningLine(THREE, color, radius, seed) {
  const points = Array.from({ length: 14 }, () => new THREE.Vector3());
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity: 0,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const line = new THREE.Line(geometry, material);
  line.userData = { points, radius, phase: stableUnit(seed) * 9, seed, next: 0 };
  return line;
}
function refreshLightningLine(line, THREE, t) {
  const points = line.userData.points;
  const radius = line.userData.radius || 5;
  const angle = t * 0.17 + line.userData.phase;
  const from = new THREE.Vector3(Math.cos(angle) * radius, Math.sin(angle * 0.81) * radius * 0.58, -1.2 + Math.sin(angle) * 1.2);
  const to = new THREE.Vector3(0, 0, 0);
  const tick = Math.floor(t * 12) + Number(state.visualSignals?.revision || 0) * 17;
  for (let i = 0; i < points.length; i++) {
    const k = i / (points.length - 1);
    const bend = Math.sin(k * Math.PI) * 0.32;
    points[i].lerpVectors(from, to, k);
    points[i].x += (stableUnit(line.userData.seed + tick * 131 + i * 17) - 0.5) * bend;
    points[i].y += (stableUnit(line.userData.seed + tick * 137 + i * 19) - 0.5) * bend;
    points[i].z += (stableUnit(line.userData.seed + tick * 149 + i * 23) - 0.5) * bend;
  }
  line.geometry.setFromPoints(points);
}
function createSparkField(THREE, count, spread, color, size, seed) {
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const r = spread * Math.pow(stableUnit(seed + i * 7), 0.72);
    const a = stableUnit(seed + i * 7 + 1) * Math.PI * 2;
    const y = (stableUnit(seed + i * 7 + 2) - 0.5) * spread * 0.92;
    positions[i * 3] = Math.cos(a) * r;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = Math.sin(a) * r * 0.62;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color,
    size,
    transparent: true,
    opacity: 0.68,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  return new THREE.Points(geometry, material);
}
function makeTextSprite(THREE, text, color) {
  const c = document.createElement('canvas');
  c.width = 192;
  c.height = 96;
  const g = c.getContext('2d');
  g.clearRect(0, 0, c.width, c.height);
  g.font = '700 34px Inter, system-ui, sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.shadowColor = `#${color.toString(16).padStart(6, '0')}`;
  g.shadowBlur = 14;
  g.fillStyle = '#eef1f6';
  g.fillText(text, 96, 38);
  g.font = '500 13px Inter, system-ui, sans-serif';
  g.shadowBlur = 7;
  g.fillStyle = 'rgba(238,241,246,0.70)';
  g.fillText(CORE_QUBITS.find(q => q.id === text)?.role || '', 96, 65);
  const texture = new THREE.CanvasTexture(c);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    opacity: 0.82,
    depthWrite: false,
  });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.82, 0.41, 1);
  return sprite;
}
function makePanelSprite(THREE, title, detail, color, width = 320, height = 112) {
  const c = document.createElement('canvas');
  c.width = width;
  c.height = height;
  const g = c.getContext('2d');
  const hex = `#${color.toString(16).padStart(6, '0')}`;
  g.clearRect(0, 0, c.width, c.height);
  g.fillStyle = 'rgba(8,10,14,0.42)';
  g.strokeStyle = hex;
  g.lineWidth = 2;
  g.beginPath();
  g.roundRect(10, 10, c.width - 20, c.height - 20, 12);
  g.fill();
  g.stroke();
  g.font = '800 24px Inter, system-ui, sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.shadowColor = hex;
  g.shadowBlur = 16;
  g.fillStyle = '#eef1f6';
  g.fillText(title, c.width / 2, 43);
  g.font = '500 14px Inter, system-ui, sans-serif';
  g.shadowBlur = 8;
  g.fillStyle = 'rgba(238,241,246,0.68)';
  g.fillText(detail, c.width / 2, 74);
  const texture = new THREE.CanvasTexture(c);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    opacity: 0.72,
    depthWrite: false,
  });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(width / 250, height / 250, 1);
  return sprite;
}
function makeLineSegments(THREE, pairs, positions, color, opacity = 0.5, dashed = false) {
  const points = [];
  for (const [a, b] of pairs) {
    points.push(positions[a].x, positions[a].y, positions[a].z, positions[b].x, positions[b].y, positions[b].z);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const line = new THREE.LineSegments(geometry, material);
  if (dashed) line.userData.dashed = true;
  return line;
}
function nearestPairs(positions, count = 3, maxPairs = 0) {
  const seen = new Set();
  const pairs = [];
  for (let i = 0; i < positions.length; i++) {
    const nearest = positions
      .map((p, j) => ({ j, d: i === j ? Infinity : positions[i].distanceToSquared(p) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, count);
    for (const n of nearest) {
      const a = Math.min(i, n.j);
      const b = Math.max(i, n.j);
      const key = `${a}:${b}`;
      if (seen.has(key)) continue;
      seen.add(key);
      pairs.push([a, b]);
      if (maxPairs && pairs.length >= maxPairs) return pairs;
    }
  }
  return pairs;
}
function vectorFromArray(THREE, value) {
  return new THREE.Vector3(value[0], value[1], value[2]);
}
function makeCurveLine(THREE, points, color, opacity = 0.34) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: true,
    opacity,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  return new THREE.Line(geometry, material);
}
function arkCurvePoints(THREE, start, end, lift = 0.72, segments = 28) {
  const mid = start.clone().lerp(end, 0.5);
  const outward = mid.lengthSq() > 0.001 ? mid.clone().normalize() : new THREE.Vector3(0, 0, 1);
  const control = mid.add(outward.multiplyScalar(lift));
  const curve = new THREE.QuadraticBezierCurve3(start, control, end);
  return curve.getPoints(segments);
}
function createQuantumArchitectureOverlays(THREE, corePositions, reservoirPositions) {
  const root = new THREE.Group();
  const channelPulse = new Float32Array(CORE_QUBITS.length);
  const modulePulse = {
    encoding: 0,
    dynamics: 0,
    entropy: 0,
    readout: 0,
  };
  const layerRings = [];
  const arkRibs = [];
  const encodingBeams = [];
  const entropyRoutes = [];
  const readoutRoutes = [];
  const dynamicsRings = [];
  const moduleLabels = [];
  const layerLabels = [];

  for (const layer of QUANTUM_ARCHITECTURE.topologicalLayers) {
    const material = new THREE.MeshBasicMaterial({
      color: layer.color,
      transparent: true,
      opacity: 0.045,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(layer.radius, 0.006, 6, state.lowPower ? 96 : 144),
      material
    );
    ring.rotation.set(layer.tilt[0], layer.tilt[1], layer.tilt[2]);
    ring.userData = { key: layer.key, baseOpacity: 0.038, rate: 0.00045 + layerRings.length * 0.00018 };
    root.add(ring);
    layerRings.push(ring);

    const label = makePanelSprite(THREE, layer.label, layer.key, layer.color, 212, 78);
    label.position.set(-4.18, 1.8 - layerLabels.length * 0.48, -0.36);
    label.scale.set(0.58, 0.22, 1);
    label.material.opacity = 0.34;
    root.add(label);
    layerLabels.push(label);
  }

  const ribCount = state.lowPower ? 5 : 8;
  for (let i = 0; i < ribCount; i++) {
    const angle = (i / ribCount) * Math.PI * 2;
    const points = [];
    for (let j = 0; j <= 42; j++) {
      const u = (j / 42) * Math.PI;
      const x = Math.cos(angle) * Math.sin(u) * 3.94;
      const y = Math.cos(u) * 3.82;
      const z = Math.sin(angle) * Math.sin(u) * 2.28;
      points.push(new THREE.Vector3(x, y, z));
    }
    const rib = makeCurveLine(THREE, points, 0xffe39a, 0.105);
    rib.userData = { baseOpacity: 0.075, phase: i * 0.8 };
    root.add(rib);
    arkRibs.push(rib);
  }

  const nodeMaterial = color => new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.72,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const anchorGeometry = new THREE.SphereGeometry(0.055, state.lowPower ? 12 : 18, state.lowPower ? 8 : 12);
  for (const module of QUANTUM_ARCHITECTURE.modules) {
    const anchor = vectorFromArray(THREE, module.anchor);
    const anchorNode = new THREE.Mesh(anchorGeometry, nodeMaterial(module.color));
    anchorNode.position.copy(anchor);
    anchorNode.userData = { key: module.key, baseScale: 1 };
    root.add(anchorNode);

    const label = makePanelSprite(THREE, module.label, module.detail, module.color, 286, 92);
    label.position.copy(module.labelAnchor ? vectorFromArray(THREE, module.labelAnchor) : anchor.clone().multiplyScalar(1.03));
    label.position.z += module.key === 'readout' ? 0.34 : 0;
    label.scale.set(0.82, 0.31, 1);
    label.material.opacity = 0.42;
    label.userData = { key: module.key };
    root.add(label);
    moduleLabels.push(label);

    for (const target of module.targets) {
      const end = corePositions[target].clone().multiplyScalar(module.key === 'readout' ? 1.22 : 0.96);
      const line = makeCurveLine(THREE, arkCurvePoints(THREE, anchor, end, module.key === 'entropy' ? 0.98 : 0.66), module.color, 0.09);
      line.userData = { key: module.key, target, baseOpacity: module.key === 'dynamics' ? 0.055 : 0.075 };
      root.add(line);
      if (module.key === 'encoding') encodingBeams.push(line);
      if (module.key === 'entropy') entropyRoutes.push(line);
      if (module.key === 'readout') readoutRoutes.push(line);
    }
  }

  for (let i = 0; i < 3; i++) {
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(2.38 + i * 0.42, 0.008, 6, state.lowPower ? 84 : 128),
      new THREE.MeshBasicMaterial({
        color: i === 1 ? 0x78d88b : 0x56d7ff,
        transparent: true,
        opacity: 0.055,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
    );
    ring.rotation.set(0.78 + i * 0.34, 0.48 - i * 0.18, i * 0.62);
    ring.userData = { phase: i * 1.3, rate: 0.0012 + i * 0.00045 };
    root.add(ring);
    dynamicsRings.push(ring);
  }

  const readoutGate = new THREE.Mesh(
    new THREE.TorusGeometry(0.58, 0.014, 8, state.lowPower ? 64 : 96),
    new THREE.MeshBasicMaterial({
      color: 0xeef1f6,
      transparent: true,
      opacity: 0.20,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })
  );
  readoutGate.position.copy(vectorFromArray(THREE, QUANTUM_ARCHITECTURE.modules.find(m => m.key === 'readout').anchor));
  readoutGate.rotation.x = Math.PI * 0.5;
  root.add(readoutGate);

  const summaryLabel = makePanelSprite(THREE, QUANTUM_ARCHITECTURE.summary.label, QUANTUM_ARCHITECTURE.summary.detail, 0xffe39a, 300, 92);
  summaryLabel.position.set(0, 0.08, 3.08);
  summaryLabel.scale.set(0.96, 0.34, 1);
  summaryLabel.material.opacity = 0.54;
  root.add(summaryLabel);

  return {
    root,
    channelPulse,
    modulePulse,
    layerRings,
    layerLabels,
    arkRibs,
    encodingBeams,
    entropyRoutes,
    readoutRoutes,
    dynamicsRings,
    moduleLabels,
    readoutGate,
    summaryLabel,
  };
}
function createArchitectureManifold(THREE) {
  const root = new THREE.Group();
  const phi = (1 + Math.sqrt(5)) / 2;
  const base = [
    [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
    [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
    [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
  ];
  const corePositions = base.map(([x, y, z]) => new THREE.Vector3(x, y, z).normalize().multiplyScalar(2.05));
  const corePairs = nearestPairs(corePositions, 5);

  const coreLines = makeLineSegments(THREE, corePairs, corePositions, 0xb679ff, 0.46);
  root.add(coreLines);

  const coreNodes = [];
  const coreLights = [];
  const coreLabels = [];
  const nodeGeometry = new THREE.SphereGeometry(0.105, state.lowPower ? 14 : 24, state.lowPower ? 10 : 16);
  for (let i = 0; i < CORE_QUBITS.length; i++) {
    const q = CORE_QUBITS[i];
    const mat = new THREE.MeshStandardMaterial({
      color: q.color,
      emissive: q.color,
      emissiveIntensity: 0.42,
      roughness: 0.28,
      metalness: 0.22,
    });
    const node = new THREE.Mesh(nodeGeometry, mat);
    node.position.copy(corePositions[i]);
    node.userData = { index: i, baseScale: 1, role: q.role, id: q.id };
    root.add(node);
    coreNodes.push(node);

    const light = new THREE.PointLight(q.color, 0.35, 4.2);
    light.position.copy(corePositions[i]);
    root.add(light);
    coreLights.push(light);

    const label = makeTextSprite(THREE, q.id, q.color);
    label.position.copy(corePositions[i].clone().multiplyScalar(1.22));
    root.add(label);
    coreLabels.push(label);
  }

  const reservoirCount = 144;
  const reservoirPositions = [];
  const reservoirColors = new Float32Array(reservoirCount * 3);
  const reservoirBase = [];
  const reservoirPulse = new Float32Array(reservoirCount);
  const colorA = new THREE.Color(0x36d9ff);
  const colorB = new THREE.Color(0x6e4dff);
  for (let i = 0; i < reservoirCount; i++) {
    const k = i + 0.5;
    const y = 1 - (k / reservoirCount) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const a = i * Math.PI * (3 - Math.sqrt(5));
    const p = new THREE.Vector3(Math.cos(a) * r, y, Math.sin(a) * r).multiplyScalar(3.52);
    reservoirPositions.push(p);
    const mix = (p.x / 7.04) + 0.5;
    const c = colorA.clone().lerp(colorB, Math.max(0, Math.min(1, 1 - mix)));
    reservoirBase.push(c.clone());
    reservoirColors[i * 3] = c.r;
    reservoirColors[i * 3 + 1] = c.g;
    reservoirColors[i * 3 + 2] = c.b;
  }
  const reservoirGeometry = new THREE.BufferGeometry();
  reservoirGeometry.setAttribute('position', new THREE.Float32BufferAttribute(reservoirPositions.flatMap(p => [p.x, p.y, p.z]), 3));
  reservoirGeometry.setAttribute('color', new THREE.BufferAttribute(reservoirColors, 3));
  const reservoirMaterial = new THREE.PointsMaterial({
    size: state.lowPower ? 0.032 : 0.046,
    vertexColors: true,
    transparent: true,
    opacity: 0.76,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const reservoir = new THREE.Points(reservoirGeometry, reservoirMaterial);
  root.add(reservoir);

  const reservoirPairs = nearestPairs(reservoirPositions, 3, state.lowPower ? 170 : 260);
  const localLines = makeLineSegments(THREE, reservoirPairs, reservoirPositions, 0x31cfff, 0.16);
  root.add(localLines);

  const longPairs = [];
  for (let i = 0; i < (state.lowPower ? 20 : 42); i++) {
    const a = (i * 23) % reservoirCount;
    const b = (a + 53 + i * 7) % reservoirCount;
    longPairs.push([a, b]);
  }
  const longLines = makeLineSegments(THREE, longPairs, reservoirPositions, 0xffffff, 0.10, true);
  root.add(longLines);

  const couplingPositions = corePositions.concat(reservoirPositions);
  const couplingPairs = [];
  for (let i = 0; i < corePositions.length; i++) {
    const nearest = reservoirPositions
      .map((p, j) => ({ j, d: corePositions[i].distanceToSquared(p) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, 3);
    for (const n of nearest) couplingPairs.push([i, corePositions.length + n.j]);
  }
  const couplingLines = makeLineSegments(THREE, couplingPairs, couplingPositions, 0xffe39a, 0.18);
  root.add(couplingLines);

  const halo = new THREE.Mesh(
    new THREE.SphereGeometry(3.58, state.lowPower ? 32 : 64, state.lowPower ? 18 : 32),
    new THREE.MeshBasicMaterial({
      color: 0x2abfff,
      wireframe: true,
      transparent: true,
      opacity: 0.055,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })
  );
  root.add(halo);

  const architecture = createQuantumArchitectureOverlays(THREE, corePositions, reservoirPositions);
  root.add(architecture.root);

  return {
    root,
    corePositions,
    coreNodes,
    coreLights,
    coreLabels,
    coreLines,
    reservoir,
    reservoirPositions,
    reservoirColors,
    reservoirBase,
    reservoirPulse,
    reservoirMaterial,
    localLines,
    longLines,
    couplingLines,
    halo,
    architecture,
    firingIndex: 0,
    firingAt: 0,
  };
}
function createThreeField(THREE) {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: !state.lowPower,
    alpha: false,
    powerPreference: state.lowPower ? 'low-power' : 'high-performance',
  });
  renderer.setClearColor(0x08090c, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x08090c, 0.045);

  const camera = new THREE.PerspectiveCamera(42, Math.max(1, innerWidth / innerHeight), 0.1, 80);
  camera.position.set(0, 0, 8.2);

  const auroraUniforms = {
    u_time: { value: 0 },
    u_energy: { value: 0.15 },
    u_mouse: { value: new THREE.Vector2(0.5, 0.5) },
    u_resolution: { value: new THREE.Vector2(1, 1) },
  };
  const aurora = new THREE.Mesh(
    new THREE.PlaneGeometry(28, 18),
    new THREE.ShaderMaterial({
      depthWrite: false,
      depthTest: false,
      uniforms: auroraUniforms,
      vertexShader: `
        varying vec2 v_uv;
        void main() {
          v_uv = uv;
          gl_Position = vec4(position.xy, 0.0, 1.0);
        }
      `,
      fragmentShader: `
        precision highp float;
        varying vec2 v_uv;
        uniform vec2 u_resolution;
        uniform float u_time;
        uniform float u_energy;
        uniform vec2 u_mouse;
        float hash(vec2 p) {
          p = fract(p * vec2(127.1, 311.7));
          p += dot(p, p + 34.23);
          return fract(p.x * p.y);
        }
        float noise(vec2 p) {
          vec2 i = floor(p), f = fract(p);
          vec2 u = f * f * (3.0 - 2.0 * f);
          return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
                     mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
        }
        float fbm(vec2 p) {
          float v = 0.0, a = 0.5;
          mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
          for (int i = 0; i < 4; i++) { v += a * noise(p); p = m * p; a *= 0.5; }
          return v;
        }
        void main() {
          vec2 uv = v_uv;
          vec2 p = uv;
          p.x *= u_resolution.x / max(u_resolution.y, 1.0);
          p += (u_mouse - 0.5) * (0.08 + u_energy * 0.08);
          float t = u_time * 0.045;
          vec2 q = vec2(fbm(p + vec2(0.0, t)), fbm(p + vec2(5.2, 1.3 - t)));
          float band = fbm(p + 1.8 * q + vec2(t * 0.5, -t * 0.3));
          vec3 dark = vec3(0.031, 0.035, 0.047);
          vec3 gold = vec3(0.961, 0.769, 0.318);
          vec3 electric = vec3(0.239, 0.682, 1.000);
          vec3 rose = vec3(1.000, 0.420, 0.616);
          vec3 col = dark;
          col = mix(col, electric, smoothstep(0.48, 1.08, q.x + u_energy * 0.55) * 0.54);
          col = mix(col, gold, smoothstep(0.44, 0.95, band + u_energy * 0.34) * 0.72);
          col += rose * smoothstep(0.78, 1.15, q.y + u_energy * 0.32) * 0.12;
          float md = distance(uv, u_mouse);
          col += mix(gold, electric, 0.46) * smoothstep(0.42, 0.0, md) * (0.08 + u_energy * 0.10);
          float vig = smoothstep(1.24, 0.22, length(uv - 0.5));
          col = mix(dark, col, clamp(vig + 0.08, 0.0, 1.0));
          gl_FragColor = vec4(col, 1.0);
        }
      `,
    })
  );
  aurora.renderOrder = -10;
  scene.add(aurora);

  scene.add(new THREE.HemisphereLight(0xffe39a, 0x0a1226, 0.58));
  const keyLight = new THREE.DirectionalLight(0xffd98a, 1.75);
  keyLight.position.set(-4, 5, 5);
  scene.add(keyLight);
  const rimLight = new THREE.PointLight(0x3daeff, 1.6, 18);
  rimLight.position.set(-5.5, 2.8, -3.6);
  scene.add(rimLight);
  const flashLight = new THREE.PointLight(0xffe39a, 0, 22);
  flashLight.position.set(0, 0, 3.5);
  scene.add(flashLight);

  const manifold = createArchitectureManifold(THREE);
  scene.add(manifold.root);

  const dust = createSparkField(
    THREE, state.lowPower ? 220 : 520, 7.6, 0x3daeff, state.lowPower ? 0.018 : 0.024, 0x51a7,
  );
  scene.add(dust);
  const goldDust = createSparkField(
    THREE, state.lowPower ? 80 : 160, 5.0, 0xffe39a, state.lowPower ? 0.025 : 0.032, 0xa11e,
  );
  scene.add(goldDust);

  const bolts = [
    createLightningLine(THREE, 0xbff7ff, 5.7, 0x10f1),
    createLightningLine(THREE, 0xffe39a, 4.9, 0x20f2),
    createLightningLine(THREE, 0x8ecfff, 6.4, 0x30f3),
  ];
  for (const bolt of bolts) scene.add(bolt);

  state.scene3d = {
    THREE,
    renderer,
    scene,
    camera,
    auroraUniforms,
    manifold,
    dust,
    goldDust,
    bolts,
    flashLight,
    lastBoltAt: 0,
  };
  state.visualMode = 'three';
}
function resizeThreeField() {
  const s = state.scene3d;
  if (!s) return;
  configurePerformance();
  const dpr = effectiveDpr(state.render);
  state.dpr = dpr;
  s.renderer.setPixelRatio(dpr);
  s.renderer.setSize(innerWidth, innerHeight, false);
  s.camera.aspect = Math.max(0.1, innerWidth / Math.max(1, innerHeight));
  s.camera.updateProjectionMatrix();
  s.auroraUniforms.u_resolution.value.set(Math.max(1, innerWidth * dpr), Math.max(1, innerHeight * dpr));
}
function resize() {
  configurePerformance();
  if (state.scene3d) {
    resizeThreeField();
    return;
  }
  state.dpr = effectiveDpr(state.render);
  state.w = Math.floor(innerWidth * state.dpr);
  state.h = Math.floor(innerHeight * state.dpr);
  canvas.width = state.w;
  canvas.height = state.h;
  seed();
}
function seed() {
  const area = innerWidth * innerHeight;
  const divisor = state.reducedMotion ? 52000 : state.lowPower ? 28000 : 17000;
  const min = state.reducedMotion ? 24 : state.lowPower ? 44 : 72;
  const max = state.reducedMotion ? 44 : state.lowPower ? 88 : 140;
  const count = Math.min(max, Math.max(min, Math.floor(area / divisor)));
  state.particles = Array.from({ length: count }, (_, i) => ({
    a: (i / count) * Math.PI * 2,
    r: 0.12 + stableUnit(0x901 + i * 11) * 0.48,
    s: 0.0015 + stableUnit(0x902 + i * 13) * 0.004,
    z: 0.25 + stableUnit(0x903 + i * 17) * 0.75,
    hue: stableUnit(0x904 + i * 19) < 0.62 ? 42 : 214,
  }));
}
function queueResize() {
  if (resizeRaf) return;
  resizeRaf = requestAnimationFrame(() => {
    resizeRaf = 0;
    resize();
  });
}
function startDraw() {
  if (!state.raf && !document.hidden && !state.fieldHidden) state.raf = requestAnimationFrame(draw);
}
function stopDraw() {
  if (state.raf) cancelAnimationFrame(state.raf);
  state.raf = 0;
}
function drawThree(now = 0) {
  state.raf = 0;
  if (document.hidden) return;
  const frameMs = 1000 / state.targetFps;
  if (state.lastFrame && now - state.lastFrame < frameMs) {
    recordSkippedFrame(state.render);
    startDraw();
    return;
  }
  const workStartedAt = performance.now();
  const dt = Math.min(2, Math.max(0.35, (now - (state.lastFrame || now)) / 16.67 || 1));
  state.lastFrame = now;
  state.phase += dt / 60;
  state.pulse *= Math.pow(0.982, dt);
  state.pointerX += (state.targetPointerX - state.pointerX) * 0.055;
  state.pointerY += (state.targetPointerY - state.pointerY) * 0.055;

  const s = state.scene3d;
  const t = now * 0.001;
  const motionScale = state.reducedMotion ? 0.12 : 1;
  const signals = deriveVisualSignals(state.lastState, {
    live: state.realtime.live,
    connecting: state.realtime.connecting,
    speaking: state.speaking,
  });
  state.visualSignals = signals;
  const targetEnergy = Math.min(1.8, signals.energy + state.pulse * 0.62);
  state.visualEnergy += (targetEnergy - state.visualEnergy) * 0.075;
  const energy = state.visualEnergy;

  s.auroraUniforms.u_time.value = t;
  s.auroraUniforms.u_energy.value = energy;
  s.auroraUniforms.u_mouse.value.set(0.5 + state.pointerX * 0.28, 0.5 - state.pointerY * 0.28);

  const m = s.manifold;
  const narrow = innerWidth < 560;
  const compact = innerWidth < 900;
  const layoutScale = narrow ? 0.66 : compact ? 0.84 : 1;
  const breathe = 1 + Math.sin(t * 1.45) * 0.018 * motionScale + energy * 0.035;
  m.root.scale.setScalar(layoutScale * breathe);
  m.root.rotation.y += (0.0028 + energy * 0.0038) * dt * motionScale;
  m.root.rotation.x = Math.sin(t * 0.46) * 0.07 * motionScale + state.pointerY * 0.10 * motionScale;
  m.root.rotation.z = state.pointerX * 0.05;
  m.halo.rotation.y -= (0.002 + energy * 0.0018) * dt * motionScale;
  m.halo.rotation.x += 0.001 * dt * motionScale;
  m.halo.material.opacity = Math.min(0.12, 0.045 + energy * 0.026);
  m.coreLines.material.opacity = Math.min(0.72, 0.28 + energy * 0.18);
  m.localLines.material.opacity = Math.min(0.30, 0.10 + energy * 0.07);
  m.longLines.material.opacity = Math.min(0.24, 0.06 + energy * 0.09);
  m.couplingLines.material.opacity = Math.min(0.42, 0.12 + energy * 0.15);

  const shouldFire = now - m.firingAt > Math.max(90, (410 / signals.cadence) * (state.reducedMotion ? 4 : 1));
  if (shouldFire) {
    m.firingAt = now;
    m.firingIndex = (m.firingIndex + 1 + Math.floor(energy * 3)) % m.coreNodes.length;
    const current = m.coreNodes[m.firingIndex];
    current.userData.fire = 1.35 + energy * 0.6;
    if (m.architecture) {
      const arch = m.architecture;
      const entropyTargets = QUANTUM_ARCHITECTURE.modules.find(module => module.key === 'entropy').targets;
      const readoutTargets = QUANTUM_ARCHITECTURE.modules.find(module => module.key === 'readout').targets;
      arch.channelPulse[m.firingIndex] = Math.max(arch.channelPulse[m.firingIndex], 1.18 + energy * 0.36);
      arch.modulePulse.encoding = Math.max(arch.modulePulse.encoding, 0.62 + energy * 0.18);
      arch.modulePulse.dynamics = Math.max(arch.modulePulse.dynamics, 0.42 + energy * 0.24);
      if (entropyTargets.includes(m.firingIndex)) {
        arch.modulePulse.entropy = Math.max(arch.modulePulse.entropy, 0.92 + energy * 0.26);
      }
      if (readoutTargets.includes(m.firingIndex) || state.speaking || state.realtime.live) {
        arch.modulePulse.readout = Math.max(arch.modulePulse.readout, 0.82 + energy * 0.24);
      }
    }
    for (let j = 0; j < 8 + Math.floor(energy * 12); j++) {
      const idx = (m.firingIndex * 17 + j * 23 + Math.floor(t * 11)) % m.reservoirPulse.length;
      m.reservoirPulse[idx] = Math.max(m.reservoirPulse[idx], 0.85 + energy * 0.28);
    }
  }
  for (let i = 0; i < m.coreNodes.length; i++) {
    const node = m.coreNodes[i];
    const fire = node.userData.fire || 0;
    node.userData.fire = Math.max(0, fire - 0.045 * dt);
    const wave = 0.5 + 0.5 * Math.sin(t * 2.4 + i * 0.72);
    const active = i === m.firingIndex ? 0.35 : 0;
    const glow = Math.min(2.4, 0.32 + energy * 0.42 + fire * 1.15 + active);
    node.scale.setScalar(1 + fire * 0.72 + energy * 0.08 + wave * 0.035);
    node.material.emissiveIntensity = glow;
    m.coreLights[i].intensity = Math.min(4.2, 0.18 + fire * 3.1 + energy * 0.65);
    m.coreLights[i].distance = 3.4 + energy * 2.2;
    if (m.coreLabels[i]) {
      const labelScale = narrow ? 0.62 : compact ? 0.76 : 1;
      m.coreLabels[i].scale.set(0.82 * labelScale, 0.41 * labelScale, 1);
    }
  }
  if (m.architecture) {
    const arch = m.architecture;
    arch.modulePulse.encoding = Math.max(arch.modulePulse.encoding, signals.encoding);
    arch.modulePulse.dynamics = Math.max(arch.modulePulse.dynamics, signals.dynamics);
    arch.modulePulse.entropy = Math.max(arch.modulePulse.entropy, signals.entropy);
    arch.modulePulse.readout = Math.max(arch.modulePulse.readout, signals.readout);
    for (const key of Object.keys(arch.modulePulse)) {
      arch.modulePulse[key] = Math.max(0, arch.modulePulse[key] - 0.025 * dt);
    }
    for (let i = 0; i < arch.channelPulse.length; i++) {
      arch.channelPulse[i] = Math.max(0, arch.channelPulse[i] - 0.032 * dt);
    }

    arch.root.rotation.y = Math.sin(t * 0.18) * 0.04 * motionScale;
    arch.summaryLabel.material.opacity = Math.min(0.86, 0.44 + energy * 0.16 + arch.modulePulse.readout * 0.16);
    arch.summaryLabel.scale.set(0.96 + energy * 0.035, 0.34 + energy * 0.012, 1);
    arch.readoutGate.material.opacity = Math.min(0.64, 0.16 + arch.modulePulse.readout * 0.36 + energy * 0.06);
    arch.readoutGate.scale.setScalar(1 + arch.modulePulse.readout * 0.18 + Math.sin(t * 3.1) * 0.018 * motionScale);

    for (let i = 0; i < arch.layerRings.length; i++) {
      const ring = arch.layerRings[i];
      ring.rotation.z += (ring.userData.rate + energy * 0.00045) * dt * motionScale * (i % 2 ? -1 : 1);
      ring.material.opacity = Math.min(0.16, ring.userData.baseOpacity + energy * 0.028 + arch.modulePulse.dynamics * 0.018);
    }
    for (let i = 0; i < arch.arkRibs.length; i++) {
      const rib = arch.arkRibs[i];
      const wave = Math.max(0, Math.sin(t * 1.7 + rib.userData.phase));
      rib.material.opacity = Math.min(0.26, rib.userData.baseOpacity + energy * 0.045 + wave * 0.035);
    }
    for (const label of arch.layerLabels) {
      label.material.opacity = narrow ? 0.12 : Math.min(0.42, 0.22 + energy * 0.08);
    }
    for (const label of arch.moduleLabels) {
      const pulse = arch.modulePulse[label.userData.key] || 0;
      label.material.opacity = Math.min(narrow ? 0.46 : 0.72, 0.30 + pulse * 0.32 + energy * 0.06);
    }
    for (const line of arch.encodingBeams) {
      const pulse = arch.channelPulse[line.userData.target] || 0;
      line.material.opacity = Math.min(0.52, line.userData.baseOpacity + pulse * 0.30 + arch.modulePulse.encoding * 0.10);
    }
    for (const line of arch.entropyRoutes) {
      const pulse = arch.channelPulse[line.userData.target] || 0;
      line.material.opacity = Math.min(0.58, line.userData.baseOpacity + pulse * 0.14 + arch.modulePulse.entropy * 0.34 + energy * 0.04);
    }
    for (const line of arch.readoutRoutes) {
      const pulse = arch.channelPulse[line.userData.target] || 0;
      line.material.opacity = Math.min(0.62, line.userData.baseOpacity + pulse * 0.12 + arch.modulePulse.readout * 0.38);
    }
    for (let i = 0; i < arch.dynamicsRings.length; i++) {
      const ring = arch.dynamicsRings[i];
      ring.rotation.x += ring.userData.rate * dt * motionScale;
      ring.rotation.z -= (ring.userData.rate * 1.6 + energy * 0.0008) * dt * motionScale;
      ring.material.opacity = Math.min(0.28, 0.04 + arch.modulePulse.dynamics * 0.11 + energy * 0.04);
    }
  }
  const hot = new s.THREE.Color(0xffe39a);
  const white = new s.THREE.Color(0xeef1f6);
  let reservoirDirty = false;
  for (let i = 0; i < m.reservoirPulse.length; i++) {
    let pulse = m.reservoirPulse[i];
    const periodic = Math.max(0, Math.sin(t * 2.2 + i * 0.37 + m.firingIndex * 0.8) - 0.94) * (0.25 + energy * 0.25);
    pulse = Math.max(0, pulse - 0.028 * dt);
    m.reservoirPulse[i] = pulse;
    const c = m.reservoirBase[i].clone();
    if (pulse > 0.01 || periodic > 0.01) {
      const mix = Math.min(1, pulse + periodic);
      c.lerp(mix > 0.72 ? white : hot, mix);
      reservoirDirty = true;
    }
    m.reservoirColors[i * 3] = c.r;
    m.reservoirColors[i * 3 + 1] = c.g;
    m.reservoirColors[i * 3 + 2] = c.b;
  }
  if (reservoirDirty) m.reservoir.geometry.attributes.color.needsUpdate = true;
  m.reservoirMaterial.size = (state.lowPower ? 0.032 : 0.046) * (1 + energy * 0.35);
  s.dust.rotation.y += 0.0008 * dt * motionScale;
  s.dust.rotation.x = Math.sin(t * 0.21) * 0.06 * motionScale;
  s.goldDust.rotation.y -= 0.0016 * dt * motionScale;
  s.dust.material.opacity = Math.min(0.78, 0.34 + energy * 0.16);
  s.goldDust.material.opacity = Math.min(0.82, 0.38 + energy * 0.18);
  s.flashLight.intensity += (energy * 1.8 - s.flashLight.intensity) * 0.06;

  const boltDue = (energy > 0.62 || signals.urgency > 0.55)
    && now - s.lastBoltAt > (signals.speaking ? 180 : 620);
  if (boltDue || (!s.lastBoltAt && now > 300)) {
    s.lastBoltAt = now;
    const boltIndex = (signals.revision + Math.floor(t * 2) + Math.round(signals.urgency * 7))
      % s.bolts.length;
    const bolt = s.bolts[boltIndex];
    refreshLightningLine(bolt, s.THREE, t);
    bolt.material.opacity = Math.min(0.72, 0.24 + energy * 0.28);
  }
  for (const bolt of s.bolts) {
    bolt.material.opacity *= Math.pow(0.90, dt);
  }

  s.camera.position.x += (state.pointerX * 1.4 - s.camera.position.x) * 0.035;
  s.camera.position.y += (-state.pointerY * 0.9 - s.camera.position.y) * 0.035;
  const cameraZ = narrow ? 12.2 : compact ? 9.8 : state.lowPower ? 8.8 : 8.1;
  s.camera.position.z += (cameraZ - s.camera.position.z) * 0.025;
  s.camera.lookAt(0, 0, 0);
  s.renderer.render(s.scene, s.camera);
  if (recordRenderedFrame(
    state.render, now, performance.now() - workStartedAt, state.targetFps,
  )) resizeThreeField();
  startDraw();
}
function draw(now = 0) {
  if (state.scene3d) {
    drawThree(now);
    return;
  }
  state.raf = 0;
  if (document.hidden) return;
  if (!ctx) return;
  const frameMs = 1000 / state.targetFps;
  if (state.lastFrame && now - state.lastFrame < frameMs) {
    recordSkippedFrame(state.render);
    startDraw();
    return;
  }
  const workStartedAt = performance.now();
  const dt = Math.min(2, Math.max(0.5, (now - (state.lastFrame || now)) / 16.67 || 1));
  state.lastFrame = now;
  state.phase += dt / 60;
  state.pulse *= Math.pow(0.985, dt);
  const signals = deriveVisualSignals(state.lastState, {
    live: state.realtime.live,
    connecting: state.realtime.connecting,
    speaking: state.speaking,
  });
  state.visualSignals = signals;
  const targetEnergy = Math.min(1.8, signals.energy + state.pulse * 0.62);
  state.visualEnergy += (targetEnergy - state.visualEnergy) * 0.075;
  const w = state.w, h = state.h;
  ctx.fillStyle = '#06070a';
  ctx.fillRect(0, 0, w, h);
  const cx = w * 0.5;
  const cy = h * 0.48;
  const glow = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(w, h) * 0.58);
  glow.addColorStop(0, `rgba(225,186,88,${0.10 + state.visualEnergy * 0.08})`);
  glow.addColorStop(0.42, 'rgba(24,35,55,0.34)');
  glow.addColorStop(1, 'rgba(6,7,10,1)');
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, w, h);
  ctx.save();
  ctx.translate(cx, cy);
  for (const p of state.particles) {
    p.a += p.s * dt * signals.cadence * (state.reducedMotion ? 0.12 : 1);
    const breath = Math.sin(state.phase * 0.8 + p.z * 5) * 0.04;
    const rr = Math.min(w, h) * (p.r + breath + state.pulse * 0.06);
    const x = Math.cos(p.a) * rr * (1.25 - p.z * 0.25);
    const y = Math.sin(p.a * 0.72) * rr * 0.62;
    const size = (1.2 + p.z * 3.6 + state.pulse * 3.5) * state.dpr;
    ctx.beginPath();
    ctx.fillStyle = `hsla(${p.hue}, ${p.hue === 42 ? 72 : 88}%, ${58 + p.z * 18}%, ${0.18 + p.z * 0.48})`;
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.strokeStyle = `rgba(225,186,88,${0.10 + state.visualEnergy * 0.12})`;
  ctx.lineWidth = 1 * state.dpr;
  const rings = state.reducedMotion ? 1 : state.lowPower ? 2 : 4;
  for (let i = 0; i < rings; i++) {
    const r = Math.min(w, h) * (0.16 + i * 0.095 + state.pulse * 0.03);
    ctx.beginPath();
    ctx.ellipse(0, 0, r * 1.75, r * 0.52, state.phase * 0.05 + i * 0.28, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
  if (recordRenderedFrame(
    state.render, now, performance.now() - workStartedAt, state.targetFps,
  )) resize();
  startDraw();
}

function visualAudit() {
  return buildVisualAudit(state, {
    cortexLocked: !Boolean(key()),
    fallback2d: Boolean(ctx),
    width: canvas.width,
    height: canvas.height,
    architecture: state.scene3d?.manifold?.architecture ? {
      layers: state.scene3d.manifold.architecture.layerRings.length,
      labels: state.scene3d.manifold.architecture.moduleLabels.length,
      encodingBeams: state.scene3d.manifold.architecture.encodingBeams.length,
      entropyRoutes: state.scene3d.manifold.architecture.entropyRoutes.length,
      readoutRoutes: state.scene3d.manifold.architecture.readoutRoutes.length,
      dynamicsRings: state.scene3d.manifold.architecture.dynamicsRings.length,
      summary: QUANTUM_ARCHITECTURE.summary.label,
    } : null,
  });
}

export { initReactiveField, queueResize, startDraw, stopDraw, visualAudit };
