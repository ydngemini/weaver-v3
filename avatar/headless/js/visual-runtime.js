// Deterministic projection of Weaver's verified public state into visual motion.
const ACTIVE_PHASES = new Set(['accepted', 'queued', 'thinking', 'synthesizing']);
const PHASE_ENERGY = Object.freeze({
  idle: 0,
  accepted: 0.18,
  queued: 0.12,
  thinking: 0.30,
  synthesizing: 0.38,
  completed: 0.08,
  cancelled: 0.02,
  failed: 0.16,
});

function clamp(value, minimum = 0, maximum = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.min(maximum, Math.max(minimum, number)) : minimum;
}

function stableUnit(seed) {
  let value = (Number(seed) || 0) | 0;
  value = Math.imul(value ^ (value >>> 16), 0x45d9f3b);
  value = Math.imul(value ^ (value >>> 16), 0x45d9f3b);
  value ^= value >>> 16;
  return (value >>> 0) / 4294967296;
}

function laneTotals(lanes) {
  let active = 0;
  let queued = 0;
  for (const lane of Object.values(lanes && typeof lanes === 'object' ? lanes : {})) {
    active += Math.max(0, Number(lane?.active || 0));
    queued += Math.max(0, Number(lane?.queued || 0));
  }
  return { active, queued };
}

function deriveVisualSignals(snapshot, runtime = {}) {
  const valid = snapshot?.schema_version === 2 && Number.isInteger(snapshot?.revision);
  const awareness = valid && snapshot.awareness && typeof snapshot.awareness === 'object'
    ? snapshot.awareness : {};
  const cognition = valid && snapshot.cognition && typeof snapshot.cognition === 'object'
    ? snapshot.cognition : {};
  const fabric = valid && snapshot.fabric && typeof snapshot.fabric === 'object'
    ? snapshot.fabric : {};
  const voice = valid && snapshot.voice && typeof snapshot.voice === 'object'
    ? snapshot.voice : {};
  const system = valid && snapshot.system && typeof snapshot.system === 'object'
    ? snapshot.system : {};
  const freshness = valid && snapshot.freshness && typeof snapshot.freshness === 'object'
    ? snapshot.freshness : {};
  const phase = PHASE_ENERGY[cognition.phase] === undefined ? 'idle' : cognition.phase;
  const awarenessStatus = ['nominal', 'limited', 'degraded', 'no-data'].includes(awareness.status)
    ? awareness.status : 'no-data';
  const fabricStatus = ['nominal', 'watch', 'guarded'].includes(fabric.status)
    ? fabric.status : 'guarded';
  const voiceStatus = ['ready', 'warming', 'degraded', 'no-data'].includes(voice.status)
    ? voice.status : 'no-data';
  const confidence = clamp(awareness.confidence);
  const pressure = clamp(fabric.pressure);
  const lanes = laneTotals(fabric.lanes);
  const staleSources = Object.values(freshness).filter(source => source?.fresh === false).length;
  const degradedReasons = Array.isArray(awareness.degraded_reasons)
    ? Math.min(10, awareness.degraded_reasons.length) : 0;
  const realtime = Boolean(runtime.live || runtime.connecting);
  const speaking = Boolean(runtime.speaking);
  const activeCognition = ACTIVE_PHASES.has(phase);
  const phaseEnergy = PHASE_ENERGY[phase];
  const voiceEnergy = speaking ? 0.58 : realtime ? 0.32 : voiceStatus === 'warming' ? 0.08 : 0;
  const laneEnergy = Math.min(0.22, lanes.active * 0.055 + lanes.queued * 0.022);
  const healthEnergy = awarenessStatus === 'degraded' || fabricStatus === 'guarded' ? 0.16
    : awarenessStatus === 'limited' || fabricStatus === 'watch' ? 0.08 : 0;
  const energy = clamp(
    0.08 + confidence * 0.14 + phaseEnergy + voiceEnergy + laneEnergy + pressure * 0.18 + healthEnergy,
    0.06,
    1.65,
  );
  const urgency = clamp(
    pressure * 0.55 + degradedReasons * 0.05 + staleSources * 0.035
      + (fabric.ledger_valid === false ? 0.35 : 0),
  );
  return Object.freeze({
    verified: valid,
    revision: valid ? Math.max(1, snapshot.revision) : 0,
    systemReady: Boolean(system.ready),
    awarenessStatus,
    awarenessConfidence: confidence,
    phase,
    activeCognition,
    fabricStatus,
    fabricPressure: pressure,
    ledgerValid: fabric.ledger_valid === true,
    activeLanes: lanes.active,
    queuedLanes: lanes.queued,
    voiceStatus,
    realtime,
    speaking,
    staleSources,
    degradedReasons,
    energy,
    urgency,
    cadence: speaking ? 1.8 : phase === 'synthesizing' ? 1.45 : activeCognition ? 1.15 : realtime ? 0.9 : 0.62,
    readout: speaking ? 1 : phase === 'synthesizing' ? 0.78 : realtime ? 0.52 : 0.12,
    encoding: phase === 'accepted' || phase === 'queued' ? 0.86 : activeCognition ? 0.58 : 0.18,
    dynamics: clamp(0.18 + pressure * 0.55 + laneEnergy),
    entropy: fabricStatus === 'guarded' ? 0.88 : fabricStatus === 'watch' ? 0.55 : 0.24,
  });
}

function recordSkippedFrame(render) {
  render.skippedFrames += 1;
}

function recordRenderedFrame(render, frameAt, workMs, targetFps) {
  const targetMs = 1000 / Math.max(1, targetFps || 60);
  const interval = render.lastRenderedAt ? frameAt - render.lastRenderedAt : targetMs;
  render.lastRenderedAt = frameAt;
  render.renderedFrames += 1;
  render.frameEmaMs = render.frameEmaMs
    ? render.frameEmaMs * 0.92 + interval * 0.08 : interval;
  render.workEmaMs = render.workEmaMs
    ? render.workEmaMs * 0.90 + Math.max(0, workMs) * 0.10 : Math.max(0, workMs);
  const pressured = interval > targetMs * 1.32 || workMs > targetMs * 0.72;
  if (pressured) {
    render.longFrames += 1;
    render.pressureFrames += 1;
    render.recoveryFrames = Math.max(0, render.recoveryFrames - 2);
  } else {
    render.recoveryFrames += 1;
    render.pressureFrames = Math.max(0, render.pressureFrames - 1);
  }
  if (frameAt - render.lastQualityAt < 750) return false;
  if (render.pressureFrames >= 16 && render.renderScale > 0.65) {
    render.renderScale = Math.max(0.65, Math.round((render.renderScale - 0.10) * 100) / 100);
    render.pressureFrames = 0;
    render.recoveryFrames = 0;
    render.qualityChanges += 1;
    render.lastQualityAt = frameAt;
    return true;
  }
  if (render.recoveryFrames >= 240 && render.renderScale < 1) {
    render.renderScale = Math.min(1, Math.round((render.renderScale + 0.05) * 100) / 100);
    render.recoveryFrames = 0;
    render.qualityChanges += 1;
    render.lastQualityAt = frameAt;
    return true;
  }
  return false;
}

function effectiveDpr(render) {
  render.effectiveDpr = Math.max(
    0.65,
    Math.min(render.rawDpr || 1, render.dprCap || 1) * clamp(render.renderScale, 0.65, 1),
  );
  return render.effectiveDpr;
}

function buildVisualAudit(state, details = {}) {
  const signals = state.visualSignals || deriveVisualSignals(state.lastState, {
    live: state.realtime.live,
    connecting: state.realtime.connecting,
    speaking: state.speaking,
  });
  return {
    mode: state.visualMode,
    ready: state.visualReady,
    bootProgress: state.visualBootProgress,
    cortexLocked: Boolean(details.cortexLocked),
    threeReady: Boolean(state.scene3d),
    fallback2d: Boolean(details.fallback2d),
    width: Number(details.width || 0),
    height: Number(details.height || 0),
    fps: state.targetFps,
    profile: state.render.profile,
    deviceClass: state.render.deviceClass,
    iPhone16e: state.render.iPhone16e,
    rawDpr: state.render.rawDpr,
    dprCap: state.render.dprCap,
    effectiveDpr: state.render.effectiveDpr,
    renderScale: state.render.renderScale,
    frameEmaMs: Math.round(state.render.frameEmaMs * 100) / 100,
    workEmaMs: Math.round(state.render.workEmaMs * 100) / 100,
    renderedFrames: state.render.renderedFrames,
    skippedFrames: state.render.skippedFrames,
    longFrames: state.render.longFrames,
    qualityChanges: state.render.qualityChanges,
    orientation: state.render.orientation,
    viewport: {
      width: state.render.viewportWidth,
      height: state.render.viewportHeight,
      visualWidth: state.render.visualViewportWidth,
      visualHeight: state.render.visualViewportHeight,
      keyboardInset: state.render.keyboardInset,
    },
    lowPower: state.lowPower,
    reducedMotion: state.reducedMotion,
    pulse: state.pulse,
    energy: state.visualEnergy,
    pointerX: state.pointerX,
    pointerY: state.pointerY,
    semantic: {
      verified: signals.verified,
      revision: signals.revision,
      phase: signals.phase,
      activeCognition: signals.activeCognition,
      awarenessStatus: signals.awarenessStatus,
      awarenessConfidence: signals.awarenessConfidence,
      fabricStatus: signals.fabricStatus,
      fabricPressure: signals.fabricPressure,
      ledgerValid: signals.ledgerValid,
      activeLanes: signals.activeLanes,
      queuedLanes: signals.queuedLanes,
      voiceStatus: signals.voiceStatus,
      staleSources: signals.staleSources,
      urgency: signals.urgency,
    },
    lastError: state.lastVisualError || '',
    architecture: details.architecture || null,
  };
}

export {
  clamp, stableUnit, deriveVisualSignals,
  recordSkippedFrame, recordRenderedFrame, effectiveDpr, buildVisualAudit,
};
