import { ui, state, showCortexLocked, voiceStatus } from './core.js';
import { ensureSession, authAudit } from './session.js';
import {
  unlockAudio, replayPendingVoice, startRealtimeVoice, stopRealtimeVoice, startMic,
  canUseBrowserSpeech, speak, audioAudit, realtimeVoiceAudit,
  interruptVoice, replayLastVoice, openVoiceTray, collapseVoiceTray,
} from './voice.js';
import {
  pollState, schedulePoll, checkModels, chat, triggerDream,
  cancelActiveTurn, retryLastTurn, copyLastReply, conversationAudit,
} from './cortex.js';
import { initReactiveField, queueResize, startDraw, stopDraw, visualAudit } from './visualization.js';
import { initLifecycle, lifecycleAudit } from './lifecycle.js';
import {
  initAccessibility, setDiagnosticsOpen, handleDiagnosticsKeydown,
  refreshOperatorDiagnostics, accessibilityAudit,
} from './accessibility.js';
import { startStateChannel, stopStateChannel, stateChannelAudit } from './state-channel.js';

ui.wake.addEventListener('click', async () => {
  openVoiceTray();
  if (state.realtime.live || state.realtime.connecting) {
    voiceStatus('live voice active — press Stop first', 'on');
    checkModels();
    pollState();
    return;
  }
  let hasCortex = false;
  try {
    hasCortex = await ensureSession({ interactive: true });
  } catch (error) {
    hasCortex = false;
  }
  const ok = await unlockAudio();
  ui.wake.setAttribute('aria-pressed', ok ? 'true' : 'false');
  state.wakeSpoken = true;
  if (hasCortex) void startStateChannel();
  if (await replayPendingVoice()) {
    checkModels();
    pollState();
    return;
  }
  if (hasCortex) {
    voiceStatus('testing voice', 'busy');
    speak(state.lastVoiceText || "I'm here.");
    checkModels();
    pollState();
  } else {
    voiceStatus(ok ? 'visual-only mode · audio ready' : 'visual-only mode', ok ? 'on' : '');
    showCortexLocked();
  }
  if (!ok && !canUseBrowserSpeech()) voiceStatus('audio blocked', '');
});

globalThis.__weaverHeadlessAudioAudit = () => ({
  ...audioAudit(),
  fps: state.targetFps,
  particleCount: state.particles.length,
  lowPower: state.lowPower,
  reducedMotion: state.reducedMotion,
  visualMode: state.visualMode,
  threeReady: Boolean(state.scene3d),
  visualEnergy: state.visualEnergy,
  lastVisualError: state.lastVisualError || '',
});
globalThis.__weaverHeadlessVisualAudit = visualAudit;
globalThis.__weaverRealtimeVoiceAudit = realtimeVoiceAudit;
globalThis.__weaverHeadlessSessionAudit = authAudit;
globalThis.__weaverHeadlessConversationAudit = conversationAudit;
globalThis.__weaverHeadlessLifecycleAudit = lifecycleAudit;
globalThis.__weaverHeadlessAccessibilityAudit = accessibilityAudit;
globalThis.__weaverHeadlessStateChannelAudit = stateChannelAudit;

ui.diagnosticsToggle.addEventListener('click', () => setDiagnosticsOpen(true));
ui.diagnosticsClose.addEventListener('click', () => setDiagnosticsOpen(false));
ui.diagnosticsScrim.addEventListener('click', () => setDiagnosticsOpen(false));
ui.reconnectNow.addEventListener('click', async () => {
  ui.reconnectNow.disabled = true;
  ui.reconnectNow.textContent = 'Reconnecting…';
  try {
    await pollState();
    await startStateChannel({ interactive: true, force: true });
    refreshOperatorDiagnostics();
  } finally {
    ui.reconnectNow.disabled = false;
    ui.reconnectNow.textContent = 'Reconnect now';
  }
});
ui.live.addEventListener('click', async () => {
  if (await startRealtimeVoice()) void pollState();
});
ui.mic.addEventListener('click', () => { void startMic(chat); });
ui.voiceReplay.addEventListener('click', () => { void replayLastVoice(); });
ui.voiceInterrupt.addEventListener('click', interruptVoice);
ui.voiceTrayClose.addEventListener('click', collapseVoiceTray);
ui.dream.addEventListener('click', triggerDream);
ui.copyLast.addEventListener('click', copyLastReply);
ui.retryTurn.addEventListener('click', retryLastTurn);
ui.stopTurn.addEventListener('click', () => cancelActiveTurn());

function resizeComposer() {
  ui.text.style.height = 'auto';
  ui.text.style.height = `${Math.min(112, Math.max(42, ui.text.scrollHeight))}px`;
}

function submitComposer() {
  const text = ui.text.value.trim();
  if (!text) return;
  ui.text.value = '';
  resizeComposer();
  void chat(text);
}

document.getElementById('bar').addEventListener('submit', event => {
  event.preventDefault();
  submitComposer();
});
ui.text.addEventListener('input', resizeComposer);
ui.text.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    submitComposer();
  }
});
addEventListener('pointermove', event => {
  state.targetPointerX = (event.clientX / Math.max(1, innerWidth) - 0.5) * 2;
  state.targetPointerY = (event.clientY / Math.max(1, innerHeight) - 0.5) * 2;
}, { passive: true });
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    stopDraw();
    stopStateChannel({ permanent: false });
    schedulePoll(20000);
  } else {
    state.lastFrame = 0;
    startDraw();
    void startStateChannel();
    schedulePoll(0);
  }
});
document.addEventListener('keydown', event => {
  if (handleDiagnosticsKeydown(event)) return;
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    ui.text.focus({ preventScroll: false });
    return;
  }
  if (event.key === 'Escape' && conversationAudit().active) {
    event.preventDefault();
    void cancelActiveTurn();
  }
});
addEventListener('weaver:network-restored', () => {
  void pollState().then(() => startStateChannel({ force: true }));
  schedulePoll(0);
});
addEventListener('weaver:network-lost', () => {
  stopStateChannel({ permanent: false });
  schedulePoll(20_000);
});
addEventListener('pagehide', () => {
  void cancelActiveTurn({ remote: false });
  stopRealtimeVoice('');
  stopStateChannel();
  stopDraw();
}, { once: true });

initAccessibility({
  onVisualPreference: () => {
    queueResize();
    if (state.fieldHidden) stopDraw();
    else startDraw();
  },
});
initLifecycle({ onViewportChange: queueResize });
initReactiveField();
void pollState().then(() => startStateChannel());
schedulePoll();
