// Shared browser state and small, side-effect-free UI helpers.
const canvas = document.getElementById('field');
const audio = document.getElementById('audio');
audio.preload = 'auto';
audio.playsInline = true;
audio.setAttribute('playsinline', '');
const ui = {
  stamp: document.getElementById('stamp'),
  brain: document.getElementById('brain'),
  voice: document.getElementById('voice'),
  thoughts: document.getElementById('thoughts'),
  dreams: document.getElementById('dreams'),
  lastThought: document.getElementById('lastThought'),
  lastDream: document.getElementById('lastDream'),
  heard: document.getElementById('heard'),
  said: document.getElementById('said'),
  brainDot: document.getElementById('brainDot'),
  voiceDot: document.getElementById('voiceDot'),
  thoughtDot: document.getElementById('thoughtDot'),
  dreamDot: document.getElementById('dreamDot'),
  wake: document.getElementById('wake'),
  live: document.getElementById('live'),
  mic: document.getElementById('mic'),
  send: document.getElementById('send'),
  dream: document.getElementById('dream'),
  text: document.getElementById('text'),
  visualBoot: document.getElementById('visualBoot'),
  visualBootTitle: document.getElementById('visualBootTitle'),
  visualBootFill: document.getElementById('visualBootFill'),
  visualBootDetail: document.getElementById('visualBootDetail'),
  transcript: document.getElementById('transcript'),
  heardTime: document.getElementById('heardTime'),
  turnStatus: document.getElementById('turnStatus'),
  turnStatusText: document.getElementById('turnStatusText'),
  turnElapsed: document.getElementById('turnElapsed'),
  reactionReadout: document.getElementById('reactionReadout'),
  copyLast: document.getElementById('copyLast'),
  retryTurn: document.getElementById('retryTurn'),
  stopTurn: document.getElementById('stopTurn'),
  connectionAnnouncement: document.getElementById('connectionAnnouncement'),
  awarenessState: document.getElementById('awarenessState'),
  awarenessConfidence: document.getElementById('awarenessConfidence'),
  awarenessConfidenceFill: document.getElementById('awarenessConfidenceFill'),
  awarenessReason: document.getElementById('awarenessReason'),
  diagnosticsToggle: document.getElementById('diagnosticsToggle'),
  diagnosticsClose: document.getElementById('diagnosticsClose'),
  diagnosticsDrawer: document.getElementById('diagnosticsDrawer'),
  diagnosticsScrim: document.getElementById('diagnosticsScrim'),
  diagnosticRevision: document.getElementById('diagnosticRevision'),
  diagnosticFreshness: document.getElementById('diagnosticFreshness'),
  diagnosticReaction: document.getElementById('diagnosticReaction'),
  diagnosticTransport: document.getElementById('diagnosticTransport'),
  voiceTray: document.getElementById('voiceTray'),
  voiceSessionTitle: document.getElementById('voiceSessionTitle'),
  voiceSessionDetail: document.getElementById('voiceSessionDetail'),
  voiceReplay: document.getElementById('voiceReplay'),
  voiceInterrupt: document.getElementById('voiceInterrupt'),
  voiceTrayClose: document.getElementById('voiceTrayClose'),
  voiceMeterFill: document.getElementById('voiceMeterFill'),
  voicePermission: document.getElementById('voicePermission'),
  voiceDevice: document.getElementById('voiceDevice'),
  voiceLatency: document.getElementById('voiceLatency'),
  voiceCaptions: document.getElementById('voiceCaptions'),
  voiceUserCaption: document.getElementById('voiceUserCaption'),
  voiceWeaverCaption: document.getElementById('voiceWeaverCaption'),
  networkPill: document.getElementById('networkPill'),
  networkDot: document.getElementById('networkDot'),
  networkStatus: document.getElementById('networkStatus'),
  installApp: document.getElementById('installApp'),
  appShell: document.querySelector('.app-shell'),
  connectionBanner: document.getElementById('connectionBanner'),
  connectionMessage: document.getElementById('connectionMessage'),
  reconnectNow: document.getElementById('reconnectNow'),
  motionToggle: document.getElementById('motionToggle'),
  motionStatus: document.getElementById('motionStatus'),
  contrastToggle: document.getElementById('contrastToggle'),
  contrastStatus: document.getElementById('contrastStatus'),
  fieldToggle: document.getElementById('fieldToggle'),
  fieldStatus: document.getElementById('fieldStatus'),
  resetPreferences: document.getElementById('resetPreferences'),
  diagnosticConnection: document.getElementById('diagnosticConnection'),
  diagnosticSession: document.getElementById('diagnosticSession'),
  diagnosticNetwork: document.getElementById('diagnosticNetwork'),
  diagnosticRender: document.getElementById('diagnosticRender'),
  diagnosticVoice: document.getElementById('diagnosticVoice'),
  diagnosticPrivacy: document.getElementById('diagnosticPrivacy'),
};
function loadEphemeralBrainKey() {
  try {
    const sessionKey = sessionStorage.getItem('weaver_llm_key') || '';
    const legacyKey = localStorage.getItem('weaver_llm_key') || '';
    localStorage.removeItem('weaver_llm_key');
    sessionStorage.removeItem('weaver_llm_key');
    return (sessionKey || legacyKey).trim().slice(0, 512);
  } catch (e) {
    return '';
  }
}
const state = {
  key: loadEphemeralBrainKey(),
  w: 1,
  h: 1,
  dpr: 1,
  phase: 0,
  pulse: 0.2,
  voiceReady: false,
  audioUnlocked: false,
  audioError: '',
  wakeSpoken: false,
  speaking: false,
  voiceMode: 'locked',
  speechRun: 0,
  voiceAbort: null,
  playReject: null,
  pendingVoiceUrl: '',
  pendingVoiceText: '',
  lastVoiceText: '',
  lastVoiceError: '',
  lastTtsStatus: 0,
  lastTtsContentType: '',
  lastPlaybackAt: 0,
  systemReducedMotion: matchMedia('(prefers-reduced-motion: reduce)').matches,
  userReducedMotion: false,
  reducedMotion: matchMedia('(prefers-reduced-motion: reduce)').matches,
  systemHighContrast: matchMedia('(prefers-contrast: more)').matches,
  userHighContrast: false,
  fieldHidden: false,
  lowPower: false,
  targetFps: 60,
  lastFrame: 0,
  raf: 0,
  particles: [],
  scene3d: null,
  visualMode: 'boot',
  visualReady: false,
  visualBootProgress: 0.08,
  pointerX: 0,
  pointerY: 0,
  targetPointerX: 0,
  targetPointerY: 0,
  visualEnergy: 0,
  visualSignals: null,
  lastState: null,
  render: {
    profile: 'balanced',
    deviceClass: 'web',
    iPhone16e: false,
    saveData: false,
    rawDpr: 1,
    dprCap: 1.5,
    renderScale: 1,
    effectiveDpr: 1,
    frameEmaMs: 0,
    workEmaMs: 0,
    renderedFrames: 0,
    skippedFrames: 0,
    longFrames: 0,
    pressureFrames: 0,
    recoveryFrames: 0,
    qualityChanges: 0,
    lastRenderedAt: 0,
    lastQualityAt: 0,
    orientation: 'portrait',
    viewportWidth: 1,
    viewportHeight: 1,
    visualViewportWidth: 1,
    visualViewportHeight: 1,
    keyboardInset: 0,
  },
  lifecycle: {
    online: navigator.onLine,
    serviceWorker: 'unsupported',
    controlled: false,
    installAvailable: false,
    installed: false,
    displayMode: 'browser',
  },
  connection: {
    status: 'starting',
    message: '',
    failures: 0,
    lastChangedAt: Date.now(),
    lastAnnouncement: '',
  },
  channel: {
    ws: null,
    status: 'idle',
    attempt: 0,
    reconnects: 0,
    reconnectTimer: 0,
    watchdogTimer: 0,
    heartbeatIntervalMs: 10_000,
    lastMessageAt: 0,
    lastHeartbeatAt: 0,
    lastRevision: 0,
    lastError: '',
    capsuleReceipts: 0,
    intentionalStop: false,
  },
  auth: {
    status: 'locked',
    csrfToken: '',
    expiresAt: 0,
    renewTimer: 0,
    generation: 0,
    lastError: '',
  },
  nativeShell: new URLSearchParams(location.search).get('nativeShell') === '1',
  realtime: {
    ws: null,
    stream: null,
    source: null,
    processor: null,
    inputRate: 0,
    outputRate: 24000,
    playHead: 0,
    live: false,
    connecting: false,
    startedAt: 0,
    bytesSent: 0,
    bytesReceived: 0,
    framesSent: 0,
    framesReceived: 0,
    droppedFrames: 0,
    reconnects: 0,
    lastError: '',
    lastStatus: '',
    lastHeard: '',
    lastSaid: '',
    lastRoute: null,
    cortexRouted: false,
    lastReactionMs: null,
    lastSemanticLatencyMs: null,
    reactionTargetMs: 200,
    lastAudioAt: 0,
    model: 'amazon.nova-2-sonic-v1:0',
    voiceId: 'tiffany',
    mode: '',
    stopTimer: 0,
    runId: 0,
    activeSources: [],
    permission: 'prompt',
    deviceLabel: '',
    inputLevel: 0,
    outputLevel: 0,
    protocolVersion: 2,
    nextInputSequence: 1,
    lastOutputSequence: 0,
    lastInputAckSequence: 0,
    resumeToken: '',
    intentionalStop: false,
    reconnectAttempt: 0,
    reconnectTimer: 0,
    pingTimer: 0,
    telemetryTimer: 0,
    renewalTimer: 0,
    lastPongAt: 0,
    permissionCheckedAt: 0,
  },
};
const HEADLESS_MODEL = 'weaver-one';
const HEADLESS_MODEL_LABEL = 'full-stack cortex';
const THREE_MODULE_URL = '/vendor/three.module.js';
function key() {
  return state.key;
}
function requestBrainKey() {
  if (state.key) return state.key;
  const entered = (prompt('Enter Weaver brain key once to open a secure browser session, or cancel for visual-only mode:') || '').trim().slice(0, 512);
  if (!entered) return '';
  state.key = entered;
  return state.key;
}
function clearBrainKey() {
  state.key = '';
  try {
    localStorage.removeItem('weaver_llm_key');
    sessionStorage.removeItem('weaver_llm_key');
  } catch (e) {}
}
function headers(json = false) {
  const brainKey = key();
  if (!brainKey) {
    const error = new Error('cortex locked');
    error.code = 'WEAVER_CORTEX_LOCKED';
    throw error;
  }
  const h = { 'X-Weaver-Key': brainKey };
  if (json) h['Content-Type'] = 'application/json';
  return h;
}
function setVisualBoot(progress, detail, ready = false) {
  state.visualBootProgress = Math.max(state.visualBootProgress, Math.min(1, Number(progress) || 0));
  if (ui.visualBootFill) ui.visualBootFill.style.width = `${Math.round(state.visualBootProgress * 100)}%`;
  if (ui.visualBootDetail && detail) ui.visualBootDetail.textContent = detail;
  if (!ready) {
    if (ui.stamp.textContent === 'starting') ui.stamp.textContent = 'loading visual';
    return;
  }
  state.visualReady = true;
  state.visualBootProgress = 1;
  if (ui.visualBootFill) ui.visualBootFill.style.width = '100%';
  if (ui.visualBootTitle) ui.visualBootTitle.textContent = 'WEAVER IS READY';
  if (ui.visualBoot) ui.visualBoot.classList.add('ready');
  if (/^(starting|loading visual)$/.test(ui.stamp.textContent)) ui.stamp.textContent = 'visual ready';
}
function showCortexLocked() {
  ui.brain.textContent = 'cortex locked · Wake to connect';
  dot(ui.brainDot, '');
}
function compact(value, max = 280) {
  return String(value || '').replace(/\s+/g, ' ').trim().slice(0, max);
}
function configurePerformance() {
  const cores = navigator.hardwareConcurrency || 4;
  const memory = navigator.deviceMemory || 4;
  const screenWidth = Number(globalThis.screen?.width || innerWidth || 1);
  const screenHeight = Number(globalThis.screen?.height || innerHeight || 1);
  const shortSide = Math.min(screenWidth, screenHeight);
  const longSide = Math.max(screenWidth, screenHeight);
  const userAgent = navigator.userAgent || '';
  const serverDevice = state.lastState?.voice?.transport?.device?.device_class;
  const iphone = /iPhone/i.test(userAgent) || serverDevice === 'iphone' || serverDevice === 'iphone-16e';
  const iPhone16e = serverDevice === 'iphone-16e'
    || (iphone && shortSide >= 385 && shortSide <= 395 && longSide >= 835 && longSide <= 855);
  const saveData = Boolean(navigator.connection?.saveData);
  const serverLowPower = state.lastState?.voice?.transport?.device?.low_power_mode === true;
  const thermal = state.lastState?.voice?.transport?.device?.thermal_state;
  const thermalPressure = thermal === 'serious' || thermal === 'critical';
  const constrainedHardware = cores <= 4 || memory <= 4;
  state.render.iPhone16e = iPhone16e;
  state.render.deviceClass = iPhone16e ? 'iphone-16e' : iphone ? 'iphone' : 'web';
  state.render.saveData = saveData;
  state.lowPower = state.reducedMotion || saveData || serverLowPower || thermalPressure
    || (constrainedHardware && !iPhone16e);
  state.targetFps = state.reducedMotion ? 18 : thermalPressure ? 24 : state.lowPower ? 30 : 60;
  state.render.profile = state.reducedMotion ? 'reduced-motion'
    : thermalPressure ? 'thermal'
      : saveData || serverLowPower ? 'efficiency'
        : iPhone16e ? 'iphone-16e-adaptive' : state.lowPower ? 'low-power' : 'quality';
  state.render.rawDpr = Math.max(1, Number(devicePixelRatio || 1));
  state.render.dprCap = state.reducedMotion ? 1
    : iPhone16e ? 1.25 : state.lowPower ? 1.05 : 1.55;
  state.render.renderScale = Math.min(1, Math.max(0.65, state.render.renderScale || 1));
  state.render.effectiveDpr = Math.max(
    0.65,
    Math.min(state.render.rawDpr, state.render.dprCap) * state.render.renderScale,
  );
  state.render.orientation = innerWidth >= innerHeight ? 'landscape' : 'portrait';
  state.render.viewportWidth = Math.max(1, innerWidth);
  state.render.viewportHeight = Math.max(1, innerHeight);
}
function dot(el, mode) {
  el.classList.toggle('on', mode === 'on');
  el.classList.toggle('busy', mode === 'busy');
}
function setConnectionState(status, message = '', { action = true } = {}) {
  const allowed = new Set(['starting', 'connected', 'limited', 'reconnecting', 'offline', 'locked', 'failed']);
  const next = allowed.has(status) ? status : 'failed';
  const safeMessage = compact(message, 180);
  const changed = state.connection.status !== next || state.connection.message !== safeMessage;
  state.connection.status = next;
  state.connection.message = safeMessage;
  if (changed) state.connection.lastChangedAt = Date.now();
  const visible = !['starting', 'connected', 'locked'].includes(next) && Boolean(safeMessage);
  if (ui.connectionBanner) {
    ui.connectionBanner.hidden = !visible;
    ui.connectionBanner.dataset.state = next;
  }
  if (ui.connectionMessage && safeMessage) ui.connectionMessage.textContent = safeMessage;
  if (ui.reconnectNow) ui.reconnectNow.hidden = !visible || !action || next === 'offline';
  if (changed && safeMessage && state.connection.lastAnnouncement !== safeMessage) {
    state.connection.lastAnnouncement = safeMessage;
    if (ui.connectionAnnouncement) ui.connectionAnnouncement.textContent = safeMessage;
  }
}
function voiceStatus(text, mode = 'on') {
  ui.voice.textContent = text;
  dot(ui.voiceDot, mode);
}
async function brainFetch(path, options = {}) {
  const r = await fetch(path, { ...options, headers: { ...headers(Boolean(options.body)), ...(options.headers || {}) } });
  if (!r.ok) {
    if (r.status === 403) {
      clearBrainKey();
    }
    const error = new Error(`${path} ${r.status}`);
    error.status = r.status;
    throw error;
  }
  return r.json();
}

export {
  canvas, audio, ui, state,
  HEADLESS_MODEL, HEADLESS_MODEL_LABEL, THREE_MODULE_URL,
  key, requestBrainKey, clearBrainKey, headers, setVisualBoot, showCortexLocked, compact,
  configurePerformance, dot, voiceStatus, brainFetch,
  setConnectionState,
};
