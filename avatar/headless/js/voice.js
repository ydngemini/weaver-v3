import {
  audio, ui, state, compact, voiceStatus,
} from './core.js';
import { ensureSession, renewSession, sessionFetch } from './session.js';
import {
  VOICE_RECONNECT_POLICY, setVoiceTray, setVoiceLevel, setVoicePermission,
  safeDeviceTelemetry, refreshVoiceDevice, checkMicrophonePermission,
  realtimeVoiceUrl, encodeVoiceFrame, base64ToBytes, downsampleToPcm16,
  reconnectDelay,
} from './voice-support.js';
const REALTIME_VOICE_INPUT_RATE = 16000;
const REALTIME_VOICE_OUTPUT_RATE = 24000;
const REALTIME_MAX_BUFFERED_BYTES = 256 * 1024;
const SILENT_WAV = 'data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAESsAACJWAAACABAAZGF0YQAAAAA=';
const TRAINED_VOICE_TIMEOUT_MS = 10000;
let audioCtx = null;
function clearPendingVoice(revoke = true) {
  if (state.pendingVoiceUrl && revoke) {
    try { URL.revokeObjectURL(state.pendingVoiceUrl); } catch (e) {}
  }
  state.pendingVoiceUrl = '';
  state.pendingVoiceText = '';
}
function setPendingVoice(url, text, error) {
  clearPendingVoice();
  state.pendingVoiceUrl = url || '';
  state.pendingVoiceText = compact(text, 520);
  state.lastVoiceError = error?.message || String(error || '');
  voiceStatus('voice ready - tap Wake', 'busy');
  ui.voiceReplay.disabled = false;
  setVoiceTray(true, 'idle', 'Response ready to replay', 'Tap Replay or Wake after audio is allowed.');
}
function canUseBrowserSpeech() {
  return 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;
}
function browserVoices() {
  return canUseBrowserSpeech() ? speechSynthesis.getVoices() : [];
}
function voiceScore(voice) {
  const haystack = `${voice.name} ${voice.voiceURI} ${voice.lang}`.toLowerCase();
  let score = 0;
  if (/^en[-_]/.test((voice.lang || '').toLowerCase())) score += 40;
  if (/aria|jenny|samantha|ava|allison|susan|victoria|karen|zira/.test(haystack)) score += 35;
  if (/google (us|uk) english|natural|neural|premium|enhanced/.test(haystack)) score += 25;
  if (/female/.test(haystack)) score += 12;
  if (/daniel|david|mark|george|male/.test(haystack)) score -= 12;
  if (voice.localService === false) score += 4;
  return score;
}
function pickBrowserVoice() {
  return browserVoices()
    .slice()
    .sort((a, b) => voiceScore(b) - voiceScore(a) || a.name.localeCompare(b.name))[0] || null;
}
function waitForBrowserVoices() {
  if (!canUseBrowserSpeech() || browserVoices().length) return Promise.resolve();
  return new Promise(resolve => {
    const done = () => {
      speechSynthesis.removeEventListener?.('voiceschanged', done);
      resolve();
    };
    speechSynthesis.addEventListener?.('voiceschanged', done, { once: true });
    setTimeout(done, 800);
  });
}
function unlockBrowserSpeech() {
  if (!canUseBrowserSpeech()) return;
  try {
    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(' ');
    utterance.volume = 0;
    utterance.rate = 1;
    speechSynthesis.speak(utterance);
  } catch (e) {}
}
function silenceCurrentSpeech() {
  if (state.voiceAbort) {
    state.voiceAbort.abort();
    state.voiceAbort = null;
  }
  if (state.playReject) {
    state.playReject(new Error('stale speech'));
    state.playReject = null;
  }
  try {
    audio.pause();
    audio.removeAttribute('src');
    audio.load();
  } catch (e) {}
  if (canUseBrowserSpeech()) {
    try { speechSynthesis.cancel(); } catch (e) {}
  }
}
async function unlockAudio() {
  if (state.audioUnlocked) return true;
  let unlocked = false;
  state.audioError = '';
  unlockBrowserSpeech();
  try {
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (AudioContextCtor) {
      audioCtx = audioCtx || new AudioContextCtor();
      if (audioCtx.state === 'suspended') await audioCtx.resume();
      const gain = audioCtx.createGain();
      gain.gain.value = 0.00001;
      const osc = audioCtx.createOscillator();
      osc.frequency.value = 220;
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.025);
      unlocked = true;
    }
  } catch (e) {
    state.audioError = e.message || String(e);
  }
  try {
    audio.muted = true;
    audio.volume = 0;
    audio.src = SILENT_WAV;
    audio.load();
    const p = audio.play();
    if (p) await p;
    audio.pause();
    audio.currentTime = 0;
    audio.removeAttribute('src');
    audio.load();
    audio.muted = false;
    audio.volume = 1;
    unlocked = true;
  } catch (e) {
    state.audioError = e.message || String(e);
    audio.muted = false;
    audio.volume = 1;
  }
  state.audioUnlocked = unlocked;
  state.voiceReady = unlocked || canUseBrowserSpeech();
  state.voiceMode = unlocked ? 'trained-ready' : (canUseBrowserSpeech() ? 'browser-ready' : 'locked');
  if (unlocked) voiceStatus('trained voice ready', 'on');
  else if (canUseBrowserSpeech()) voiceStatus('browser voice ready', 'on');
  else voiceStatus(`audio blocked: ${state.audioError || 'tap Wake again'}`, '');
  setVoiceTray(
    true,
    unlocked ? 'idle' : 'failed',
    unlocked ? 'Voice output ready' : 'Audio permission needed',
    unlocked ? 'Trained voice is ready; Live will request microphone access.' : 'Use Wake again after allowing audio playback.',
  );
  return unlocked;
}
async function replayPendingVoice() {
  const runId = state.speechRun;
  if (state.pendingVoiceUrl) {
    const url = state.pendingVoiceUrl;
    const text = state.pendingVoiceText;
    state.pendingVoiceUrl = '';
    state.pendingVoiceText = '';
    state.speaking = true;
    state.pulse = Math.max(state.pulse, 1);
    state.voiceMode = 'trained-replay';
    voiceStatus('playing voice', 'busy');
    try {
      await playVoiceUrl(url, runId, { revokeOnFailure: false });
      state.speaking = false;
      state.audioError = '';
      state.lastVoiceError = '';
      state.lastPlaybackAt = performance.now();
      voiceStatus('voice ready', 'on');
      ui.voiceReplay.disabled = !state.lastVoiceText;
      setVoiceTray(true, 'idle', 'Replay complete', 'Weaver’s trained voice is ready.');
      return true;
    } catch (e) {
      state.speaking = false;
      setPendingVoice(url, text, e);
      return false;
    }
  }
  if (state.pendingVoiceText) {
    const text = state.pendingVoiceText;
    clearPendingVoice();
    if (state.audioUnlocked) {
      await speak(text);
      return true;
    }
    if (canUseBrowserSpeech()) {
      return browserSpeak(text, false, runId);
    }
    await speak(text);
    return true;
  }
  return false;
}

function updateRealtimeButton() {
  ui.live.disabled = false;
  ui.live.textContent = state.realtime.live ? 'Stop' : (state.realtime.connecting ? 'Live...' : 'Live');
  ui.live.classList.toggle('primary', state.realtime.live);
  ui.live.setAttribute('aria-pressed', state.realtime.live ? 'true' : 'false');
}
function setRealtimeStatus(text, mode = state.realtime.live ? 'busy' : '', trayState = '') {
  state.realtime.lastStatus = compact(text, 180);
  voiceStatus(state.realtime.lastStatus || 'live voice', mode);
  const nextState = trayState || (
    state.realtime.live ? (state.speaking ? 'speaking' : 'listening')
      : (state.realtime.connecting ? 'connecting' : 'idle')
  );
  setVoiceTray(true, nextState, 'Live cortex voice', state.realtime.lastStatus || 'Voice session ready.');
  updateRealtimeButton();
}
async function playRealtimePcm(audioBase64, sampleRate = REALTIME_VOICE_OUTPUT_RATE) {
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) throw new Error('audio context unavailable');
  audioCtx = audioCtx || new AudioContextCtor();
  if (audioCtx.state === 'suspended') await audioCtx.resume();
  const bytes = base64ToBytes(audioBase64);
  const samples = Math.floor(bytes.byteLength / 2);
  if (!samples) return;
  const view = new DataView(bytes.buffer, bytes.byteOffset, samples * 2);
  const buffer = audioCtx.createBuffer(1, samples, sampleRate);
  const out = buffer.getChannelData(0);
  let energy = 0;
  for (let i = 0; i < samples; i++) {
    out[i] = view.getInt16(i * 2, true) / 32768;
    energy += out[i] * out[i];
  }
  state.realtime.outputLevel = Math.min(1, Math.sqrt(energy / Math.max(1, samples)) * 4);
  setVoiceLevel(state.realtime.outputLevel);
  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(audioCtx.destination);
  const now = audioCtx.currentTime;
  if (!state.realtime.playHead || state.realtime.playHead < now || state.realtime.playHead - now > 1.2) {
    state.realtime.playHead = now + 0.035;
  }
  const startAt = state.realtime.playHead;
  state.realtime.playHead += buffer.duration;
  state.speaking = true;
  state.pulse = Math.max(state.pulse, 1);
  clearTimeout(state.realtime.stopTimer);
  source.onended = () => {
    const idx = state.realtime.activeSources.indexOf(source);
    if (idx >= 0) state.realtime.activeSources.splice(idx, 1);
    state.realtime.stopTimer = setTimeout(() => {
      if (audioCtx && audioCtx.currentTime >= state.realtime.playHead - 0.05) {
        state.speaking = false;
        state.realtime.outputLevel = 0;
        setVoiceLevel(0);
        if (state.realtime.live) setRealtimeStatus('live listening', 'on');
      }
    }, 90);
  };
  state.realtime.activeSources.push(source);
  source.start(startAt);
  state.realtime.lastAudioAt = performance.now();
}
function cleanupRealtimeCapture() {
  if (state.realtime.processor) {
    state.realtime.processor.onaudioprocess = null;
    try { state.realtime.processor.disconnect(); } catch (e) {}
  }
  if (state.realtime.source) {
    try { state.realtime.source.disconnect(); } catch (e) {}
  }
  if (state.realtime.stream) {
    for (const track of state.realtime.stream.getTracks()) {
      try { track.stop(); } catch (e) {}
    }
  }
  state.realtime.processor = null;
  state.realtime.source = null;
  state.realtime.stream = null;
  setVoiceLevel(0);
}
function attachRealtimeCapture(stream, ws) {
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) throw new Error('audio context unavailable');
  audioCtx = audioCtx || new AudioContextCtor();
  state.realtime.stream = stream;
  state.realtime.inputRate = audioCtx.sampleRate || 48000;
  state.realtime.source = audioCtx.createMediaStreamSource(stream);
  state.realtime.processor = audioCtx.createScriptProcessor(state.lowPower ? 4096 : 2048, 1, 1);
  state.realtime.processor.onaudioprocess = event => {
    const input = event.inputBuffer.getChannelData(0);
    let energy = 0;
    for (let index = 0; index < input.length; index += 8) energy += input[index] * input[index];
    const sampled = Math.ceil(input.length / 8);
    setVoiceLevel(Math.min(1, Math.sqrt(energy / Math.max(1, sampled)) * 5));
    if (!state.realtime.live || state.speaking || ws.readyState !== WebSocket.OPEN) return;
    if (ws.bufferedAmount > REALTIME_MAX_BUFFERED_BYTES) {
      state.realtime.droppedFrames++;
      return;
    }
    const pcm = downsampleToPcm16(input, state.realtime.inputRate, REALTIME_VOICE_INPUT_RATE);
    if (!pcm.byteLength) return;
    const sequence = state.realtime.nextInputSequence++;
    const frame = encodeVoiceFrame(pcm, sequence);
    ws.send(frame);
    state.realtime.bytesSent += pcm.byteLength;
    state.realtime.framesSent++;
  };
  state.realtime.source.connect(state.realtime.processor);
  state.realtime.processor.connect(audioCtx.destination);
}
async function handleRealtimeMessage(event) {
  const data = typeof event.data === 'string' ? JSON.parse(event.data) : {};
  if (Number.isInteger(data.serverSeq) && data.serverSeq > state.realtime.lastOutputSequence) {
    state.realtime.lastOutputSequence = data.serverSeq;
    if (state.realtime.ws?.readyState === WebSocket.OPEN) {
      state.realtime.ws.send(JSON.stringify({ type: 'output_ack', serverSeq: data.serverSeq }));
    }
  }
  if (data.type === 'ready') {
    state.realtime.model = data.model || state.realtime.model;
    state.realtime.voiceId = data.voiceId || state.realtime.voiceId;
    state.realtime.outputRate = data.outputSampleRate || REALTIME_VOICE_OUTPUT_RATE;
    state.realtime.mode = data.mode || '';
    state.realtime.cortexRouted = Boolean(data.cortexRouted);
    state.realtime.protocolVersion = Number(data.protocolVersion || 2);
    state.realtime.resumeToken = compact(data.resumeToken, 256) || state.realtime.resumeToken;
    setRealtimeStatus(`authenticated · ${state.realtime.voiceId} · protocol v${state.realtime.protocolVersion}`, 'on', 'listening');
  } else if (data.type === 'session_ready') {
    state.realtime.resumeToken = compact(data.resumeToken, 256) || state.realtime.resumeToken;
    state.realtime.nextInputSequence = Math.max(1, Number(data.ackSeq || 0) + 1);
    state.realtime.reconnectAttempt = 0;
    setRealtimeStatus(data.resumed ? 'voice resumed · listening' : 'live listening', 'on', 'listening');
  } else if (data.type === 'status') {
    const status = compact(data.status, 80) || 'live voice';
    const trayState = /thinking|connecting|preparing/i.test(status) ? 'thinking' : 'listening';
    setRealtimeStatus(status, state.realtime.live ? 'busy' : '', trayState);
  } else if (data.type === 'turn_ack') {
    state.realtime.lastReactionMs = Number(data.latencyMs ?? 0);
    state.realtime.reactionTargetMs = Number(data.reactionTargetMs || 200);
    state.pulse = 1;
    ui.voiceLatency.textContent = `Reaction ${Math.round(state.realtime.lastReactionMs)} ms · target ≤${Math.round(state.realtime.reactionTargetMs)} ms`;
    ui.diagnosticReaction.textContent = `${Math.round(state.realtime.lastReactionMs)} ms voice ack`;
    setRealtimeStatus(`heard · Weaver is thinking · ${Math.round(state.realtime.lastReactionMs)} ms ack`, 'busy', 'thinking');
  } else if (data.type === 'transcript') {
    const text = compact(data.text, 360);
    if ((data.role || '').toLowerCase() === 'user') {
      state.realtime.lastHeard = text;
      ui.voiceUserCaption.textContent = text || 'Listening…';
    } else if (data.speaker === 'weaver') {
      state.realtime.lastSaid = text;
      ui.voiceWeaverCaption.textContent = text || 'Preparing speech…';
    }
  } else if (data.type === 'agent_response') {
    if (data.speaker !== 'weaver') {
      state.realtime.lastError = 'voice speaker boundary rejected';
      setRealtimeStatus('Voice response blocked by Weaver-only boundary', '', 'failed');
      return;
    }
    const text = compact(data.text, 1200);
    state.realtime.lastSaid = text;
    state.realtime.lastRoute = null;
    state.realtime.cortexRouted = true;
    state.realtime.lastSemanticLatencyMs = Number(data.latencyMs || 0);
    ui.voiceWeaverCaption.textContent = text || 'No public response.';
    ui.voiceReplay.disabled = !text;
    state.pulse = 1;
    setRealtimeStatus('full cortex · trained voice', 'busy', 'speaking');
    await speak(text, { allowDuringRealtime: true });
  } else if (data.type === 'audio') {
    state.realtime.framesReceived++;
    state.realtime.bytesReceived += Math.floor((data.audio || '').length * 0.75);
    await playRealtimePcm(data.audio, data.sampleRate || state.realtime.outputRate);
    setRealtimeStatus('live speaking', 'busy', 'speaking');
  } else if (data.type === 'input_ack') {
    state.realtime.lastInputAckSequence = Math.max(
      state.realtime.lastInputAckSequence,
      Number(data.ackSeq || 0),
    );
  } else if (data.type === 'interrupted') {
    state.speaking = false;
    setVoiceLevel(0);
    setRealtimeStatus('interrupted · listening', 'on', 'listening');
  } else if (data.type === 'pong') {
    state.realtime.lastPongAt = performance.now();
  } else if (data.type === 'renew_required') {
    state.realtime.resumeToken = compact(data.resumeToken, 256) || state.realtime.resumeToken;
    setRealtimeStatus('refreshing authenticated voice session', 'busy', 'reconnecting');
    try {
      await renewSession();
      state.realtime.ws?.close(1012, 'renew-session');
    } catch (error) {
      state.realtime.lastError = 'voice session renewal failed';
      stopRealtimeVoice(state.realtime.lastError);
    }
  } else if (data.type === 'error') {
    state.realtime.lastError = data.code === 'voice-unavailable'
      ? 'trained live voice is warming; browser voice remains available'
      : 'live voice encountered a recoverable error';
    setRealtimeStatus(state.realtime.lastError, '', 'failed');
  }
}
function clearRealtimeTimers() {
  for (const timer of ['reconnectTimer', 'pingTimer', 'telemetryTimer', 'renewalTimer', 'stopTimer']) {
    clearTimeout(state.realtime[timer]);
    clearInterval(state.realtime[timer]);
    state.realtime[timer] = 0;
  }
}

function stopScheduledVoiceOutput() {
  state.speechRun += 1;
  silenceCurrentSpeech();
  for (const source of state.realtime.activeSources.splice(0)) {
    try { source.onended = null; source.stop(); } catch (error) {}
  }
  state.realtime.playHead = 0;
  state.realtime.outputLevel = 0;
  state.speaking = false;
  setVoiceLevel(0);
}

function stopRealtimeVoice(reason = 'live stopped') {
  state.realtime.intentionalStop = true;
  state.realtime.runId += 1;
  const ws = state.realtime.ws;
  state.realtime.live = false;
  state.realtime.connecting = false;
  clearRealtimeTimers();
  cleanupRealtimeCapture();
  stopScheduledVoiceOutput();
  if (ws && ws.readyState === WebSocket.OPEN) {
    try { ws.send(JSON.stringify({ type: 'stop' })); } catch (error) {}
    try { ws.close(1000, 'stop'); } catch (error) {}
  }
  state.realtime.ws = null;
  state.realtime.reconnectAttempt = 0;
  state.realtime.resumeToken = '';
  ui.voiceInterrupt.disabled = true;
  updateRealtimeButton();
  if (reason) setRealtimeStatus(reason, reason === 'live stopped' ? 'on' : '', 'idle');
}

function scheduleVoiceReconnect(reason = 'connection interrupted') {
  if (state.realtime.intentionalStop || state.realtime.permission !== 'granted') return;
  state.realtime.reconnectAttempt += 1;
  if (state.realtime.reconnectAttempt > VOICE_RECONNECT_POLICY.maxAttempts) {
    state.realtime.lastError = 'live voice unavailable after bounded retries';
    setRealtimeStatus(state.realtime.lastError, '', 'failed');
    return;
  }
  const delay = reconnectDelay(state.realtime.reconnectAttempt);
  state.realtime.reconnects += 1;
  setRealtimeStatus(
    `${reason} · retry ${state.realtime.reconnectAttempt}/${VOICE_RECONNECT_POLICY.maxAttempts}`,
    'busy',
    'reconnecting',
  );
  clearTimeout(state.realtime.reconnectTimer);
  state.realtime.reconnectTimer = setTimeout(() => {
    void startRealtimeVoice({ reconnect: true });
  }, delay);
}

function startVoiceHeartbeats(ws) {
  clearInterval(state.realtime.pingTimer);
  clearInterval(state.realtime.telemetryTimer);
  state.realtime.pingTimer = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'ping', t: Date.now() }));
  }, 15_000);
  state.realtime.telemetryTimer = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'telemetry', sample: safeDeviceTelemetry() }));
    }
  }, 30_000);
}

async function startRealtimeVoice(options = {}) {
  const reconnect = Boolean(options?.reconnect);
  if (!reconnect && (state.realtime.live || state.realtime.connecting)) {
    stopRealtimeVoice('live stopped');
    return false;
  }
  if (state.nativeShell) {
    setVoicePermission('native');
    await refreshVoiceDevice();
    setVoiceTray(true, 'idle', 'Native voice owns capture', 'Use the SwiftUI voice control; this web surface stays render-only.');
    return false;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    setVoicePermission('unavailable');
    setRealtimeStatus('microphone unavailable · typed chat remains ready', '', 'failed');
    return false;
  }
  try {
    await ensureSession({ interactive: !reconnect });
  } catch (error) {
    setRealtimeStatus('cortex session unavailable · tap Wake to reconnect', '', 'failed');
    return false;
  }

  const permission = await checkMicrophonePermission();
  if (permission === 'denied') {
    setRealtimeStatus('microphone blocked · enable it in browser site settings', '', 'denied');
    return false;
  }
  const runId = ++state.realtime.runId;
  state.realtime.intentionalStop = false;
  state.realtime.connecting = true;
  state.realtime.lastError = '';
  if (!reconnect) {
    state.realtime.bytesSent = 0;
    state.realtime.bytesReceived = 0;
    state.realtime.framesSent = 0;
    state.realtime.framesReceived = 0;
    state.realtime.droppedFrames = 0;
    state.realtime.nextInputSequence = 1;
    state.realtime.lastInputAckSequence = 0;
    state.realtime.lastOutputSequence = 0;
    state.realtime.resumeToken = '';
    state.realtime.reconnectAttempt = 0;
  }
  state.realtime.playHead = 0;
  updateRealtimeButton();
  setRealtimeStatus(reconnect ? 'resuming authenticated voice' : 'opening authenticated live voice', 'busy', reconnect ? 'reconnecting' : 'connecting');
  await unlockAudio();
  stopScheduledVoiceOutput();
  let stream = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    setVoicePermission('granted');
    await refreshVoiceDevice(stream);
    if (runId !== state.realtime.runId) {
      for (const track of stream.getTracks()) track.stop();
      return false;
    }
    const ws = new WebSocket(realtimeVoiceUrl(), [
      'weaver-realtime',
      `weaver-csrf.${state.auth.csrfToken}`,
    ]);
    state.realtime.ws = ws;
    state.realtime.startedAt = performance.now();
    ws.binaryType = 'arraybuffer';
    ws.onopen = async () => {
      try {
        if (runId !== state.realtime.runId) {
          try { ws.close(1000, 'stale'); } catch (error) {}
          return;
        }
        if (audioCtx?.state === 'suspended') await audioCtx.resume();
        state.realtime.live = true;
        state.realtime.connecting = false;
        attachRealtimeCapture(stream, ws);
        const start = {
          type: 'start',
          protocolVersion: 2,
          inputSampleRate: REALTIME_VOICE_INPUT_RATE,
          outputSampleRate: REALTIME_VOICE_OUTPUT_RATE,
          device: safeDeviceTelemetry(),
        };
        if (state.realtime.resumeToken) start.resumeToken = state.realtime.resumeToken;
        ws.send(JSON.stringify(start));
        startVoiceHeartbeats(ws);
        ui.voiceInterrupt.disabled = false;
        setRealtimeStatus('live listening', 'on', 'listening');
      } catch (error) {
        state.realtime.lastError = 'voice initialization failed';
        try { ws.close(1011, 'initialization'); } catch (closeError) {}
      }
    };
    ws.onmessage = event => {
      if (runId !== state.realtime.runId) return;
      handleRealtimeMessage(event).catch(() => {
        if (runId !== state.realtime.runId) return;
        state.realtime.lastError = 'invalid live voice event';
        setRealtimeStatus(state.realtime.lastError, '', 'failed');
      });
    };
    ws.onerror = () => {
      if (runId !== state.realtime.runId) return;
      state.realtime.lastError = 'live voice transport interrupted';
    };
    ws.onclose = event => {
      if (runId !== state.realtime.runId) return;
      const shouldReconnect = !state.realtime.intentionalStop && event.code !== 1000;
      clearInterval(state.realtime.pingTimer);
      clearInterval(state.realtime.telemetryTimer);
      cleanupRealtimeCapture();
      state.realtime.live = false;
      state.realtime.connecting = false;
      state.realtime.ws = null;
      ui.voiceInterrupt.disabled = true;
      updateRealtimeButton();
      if (shouldReconnect) scheduleVoiceReconnect(event.code === 1012 ? 'voice session refreshing' : 'connection interrupted');
      else setRealtimeStatus('live voice ended', 'on', 'idle');
    };
    return true;
  } catch (error) {
    if (stream) for (const track of stream.getTracks()) track.stop();
    state.realtime.connecting = false;
    state.realtime.live = false;
    const denied = error?.name === 'NotAllowedError' || error?.name === 'SecurityError';
    setVoicePermission(denied ? 'denied' : 'unavailable');
    state.realtime.lastError = denied
      ? 'microphone blocked · enable it in browser site settings'
      : 'microphone could not start · typed chat remains ready';
    updateRealtimeButton();
    setRealtimeStatus(state.realtime.lastError, '', denied ? 'denied' : 'failed');
    return false;
  }
}

function interruptVoice() {
  if (state.realtime.ws?.readyState === WebSocket.OPEN) {
    try { state.realtime.ws.send(JSON.stringify({ type: 'interrupt' })); } catch (error) {}
  }
  stopScheduledVoiceOutput();
  ui.voiceInterrupt.disabled = !state.realtime.live;
  setRealtimeStatus(state.realtime.live ? 'interrupted · listening' : 'speech interrupted', 'on', state.realtime.live ? 'listening' : 'idle');
}

async function replayLastVoice() {
  setVoiceTray(true, 'speaking', 'Replaying Weaver', 'Using trained voice with browser fallback.');
  await unlockAudio();
  if (await replayPendingVoice()) return true;
  if (!state.lastVoiceText) return false;
  await speak(state.lastVoiceText, { allowDuringRealtime: true });
  return true;
}

async function browserSpeak(text, retried = false, runId = state.speechRun) {
  if (!text || !canUseBrowserSpeech()) {
    state.audioError = 'browser speech unavailable';
    voiceStatus(state.audioError, '');
    return false;
  }
  await waitForBrowserVoices();
  return new Promise(resolve => {
    const utterance = new SpeechSynthesisUtterance(text);
    const voice = pickBrowserVoice();
    let settled = false;
    if (voice) utterance.voice = voice;
    utterance.pitch = 1.04;
    utterance.rate = 0.96;
    utterance.volume = 1;
    const finish = ok => {
      if (settled) return;
      settled = true;
      if (runId !== state.speechRun) {
        resolve(false);
        return;
      }
      state.speaking = false;
      state.voiceReady = true;
      state.voiceMode = 'browser';
      setVoiceLevel(0);
      voiceStatus(ok ? `browser voice${voice ? `: ${voice.name}` : ''}` : 'browser voice blocked', ok ? 'on' : '');
      resolve(ok);
    };
    utterance.onstart = () => {
      if (runId !== state.speechRun) return;
      state.speaking = true;
      state.pulse = Math.max(state.pulse, 1);
      state.voiceMode = 'browser';
      setVoiceLevel(0.32);
      setVoiceTray(true, 'speaking', 'Weaver is speaking', 'Browser speech fallback is active.');
      voiceStatus(`browser voice${voice ? `: ${voice.name}` : ''}`, 'busy');
    };
    utterance.onend = () => finish(true);
    utterance.onerror = event => {
      state.audioError = event.error || 'browser speech failed';
      if (!retried) {
        setTimeout(() => {
          browserSpeak(text, true, runId).then(resolve);
        }, 400);
        return;
      }
      finish(false);
    };
    try {
      speechSynthesis.cancel();
      speechSynthesis.speak(utterance);
      setTimeout(() => finish(true), Math.max(5000, text.length * 95));
    } catch (e) {
      state.audioError = e.message || String(e);
      finish(false);
    }
  });
}
function splitSentences(text) {
  const raw = String(text || '').match(/[^.!?…]+[.!?…]+["')\]]*\s*|[^.!?…]+$/g) || [text];
  const parts = [];
  for (let part of raw) {
    part = compact(part, 240);
    if (!part) continue;
    if (parts.length && (part.length < 18 || parts[parts.length - 1].length < 34)) {
      parts[parts.length - 1] = `${parts[parts.length - 1]} ${part}`;
    } else {
      parts.push(part);
    }
  }
  return parts.length ? parts.slice(0, 5) : [compact(text, 240)];
}
async function fetchVoiceUrl(text, runId) {
  if (runId !== state.speechRun) throw new Error('stale speech');
  const controller = new AbortController();
  state.voiceAbort = controller;
  const timeout = setTimeout(() => controller.abort(), TRAINED_VOICE_TIMEOUT_MS);
  const r = await sessionFetch('/brain/headless/v2/voice/synth', {
    method: 'POST',
    interactive: false,
    headers: { Accept: 'audio/mpeg,audio/wav,audio/*' },
    body: JSON.stringify({ text: compact(text, 480) }),
    signal: controller.signal,
  }).finally(() => {
    clearTimeout(timeout);
    if (state.voiceAbort === controller) state.voiceAbort = null;
  });
  if (runId !== state.speechRun) throw new Error('stale speech');
  const blob = await r.blob();
  if (runId !== state.speechRun) throw new Error('stale speech');
  if (!/^audio\//i.test(blob.type || '')) throw new Error(`tts returned ${blob.type || 'non-audio'}`);
  state.lastTtsStatus = r.status;
  state.lastTtsContentType = blob.type || r.headers.get('Content-Type') || '';
  return URL.createObjectURL(blob);
}
function playVoiceUrl(url, runId, options = {}) {
  const revokeOnFailure = options.revokeOnFailure !== false;
  return new Promise((resolve, reject) => {
    let done = false;
    const finish = (ok, error) => {
      if (done) return;
      done = true;
      setVoiceLevel(0);
      if (ok || revokeOnFailure || runId !== state.speechRun) {
        URL.revokeObjectURL(url);
      }
      audio.onended = null;
      audio.onerror = null;
      if (state.playReject === reject) state.playReject = null;
      if (runId !== state.speechRun) {
        resolve();
        return;
      }
      if (ok) {
        state.audioError = '';
        state.lastVoiceError = '';
        state.lastPlaybackAt = performance.now();
        resolve();
      } else {
        reject(error || new Error('audio playback failed'));
      }
    };
    if (runId !== state.speechRun) {
      finish(true);
      return;
    }
    state.playReject = reject;
    audio.onended = () => finish(true);
    audio.onerror = () => finish(false);
    audio.src = url;
    audio.load();
    setVoiceLevel(0.42);
    const p = audio.play();
    if (p) p.catch(error => finish(false, error));
  });
}
async function speak(text, options = {}) {
  if (!text) return;
  if ((state.realtime.live || state.realtime.connecting) && !options.allowDuringRealtime) {
    // Nova Sonic owns the audio channel — never talk over the live session.
    state.lastVoiceText = compact(text, 800);
    voiceStatus('live voice active — text shown only', 'on');
    return;
  }
  state.speechRun += 1;
  const runId = state.speechRun;
  silenceCurrentSpeech();
  state.lastVoiceText = compact(text, 800);
  state.lastVoiceError = '';
  ui.voiceReplay.disabled = !state.lastVoiceText;
  ui.voiceWeaverCaption.textContent = state.lastVoiceText || 'No public response.';
  if (state.nativeShell) {
    setVoiceTray(true, 'idle', 'Native voice owns playback', 'The SwiftUI cortex bridge handles trained audio.');
    return;
  }
  if (!state.voiceReady) {
    state.pendingVoiceText = state.lastVoiceText;
    voiceStatus('response ready · tap Replay or Wake', 'busy');
    setVoiceTray(true, 'idle', 'Weaver response ready', 'Tap Replay or Wake to allow audio before playback.');
    return;
  }
  clearPendingVoice();
  if (!state.audioUnlocked && !canUseBrowserSpeech()) {
    voiceStatus('tap Wake to unlock audio', '');
    setVoiceTray(true, 'failed', 'Audio permission needed', 'Tap Wake after allowing audio playback.');
    return;
  }
  const parts = splitSentences(text);
  state.speaking = true;
  state.pulse = Math.max(state.pulse, 1);
  voiceStatus('voice starting', 'busy');
  setVoiceTray(true, 'speaking', 'Weaver is speaking', 'Trained voice with local browser fallback.');
  let nextUrl = fetchVoiceUrl(parts[0], runId);
  for (let i = 0; i < parts.length; i++) {
    let url = '';
    try {
      url = await nextUrl;
      if (runId !== state.speechRun) {
        URL.revokeObjectURL(url);
        return;
      }
      nextUrl = (i + 1 < parts.length) ? fetchVoiceUrl(parts[i + 1], runId) : null;
      state.voiceMode = 'trained';
      voiceStatus(i === 0 ? 'speaking' : `speaking ${i + 1}/${parts.length}`, 'busy');
      await playVoiceUrl(url, runId, { revokeOnFailure: false });
      state.pulse = Math.max(state.pulse, 0.55);
    } catch (e) {
      if (runId !== state.speechRun || /stale speech/i.test(e.message || '')) return;
      state.speaking = false;
      setVoiceLevel(0);
      state.audioError = e.message || String(e);
      state.lastVoiceError = state.audioError;
      if (url && /notallowed|gesture|play|interrupted|abort/i.test(state.audioError)) {
        setPendingVoice(url, parts.slice(i).join(' ') || text, e);
        return;
      }
      voiceStatus(
        /notallowed|gesture|play/i.test(state.audioError)
          ? 'audio element blocked, using browser voice'
          : 'voice provider unavailable, using browser',
        'busy'
      );
      const remaining = parts.slice(i).join(' ');
      const spoke = await browserSpeak(remaining || text, false, runId);
      if (!spoke) {
        state.pendingVoiceText = compact(remaining || text, 520);
        voiceStatus(
          /notallowed|gesture|play/i.test(state.audioError)
            ? 'tap Wake again to unlock audio'
            : state.audioError || 'voice unavailable',
          ''
        );
      }
      setVoiceTray(true, spoke ? 'idle' : 'failed', spoke ? 'Browser voice complete' : 'Voice playback unavailable', spoke ? 'Trained voice will be retried on the next response.' : 'The text response remains available.');
      return;
    }
  }
  if (runId !== state.speechRun) return;
  state.speaking = false;
  state.voiceMode = 'trained';
  voiceStatus('voice ready', 'on');
  setVoiceLevel(0);
  setVoiceTray(true, 'idle', 'Voice ready', 'Replay is available; Live starts a continuous session.');
}

async function startMic(onTranscript) {
  if (state.realtime.live || state.realtime.connecting) stopRealtimeVoice('live stopped');
  setVoiceTray(true, 'connecting', 'One-shot microphone', 'Speak once; your final caption will be sent to Weaver.');
  if (state.nativeShell) {
    setVoicePermission('native');
    setVoiceTray(true, 'idle', 'Native voice owns capture', 'Use the SwiftUI voice control instead of the web microphone.');
    return;
  }
  await checkMicrophonePermission();
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    setVoicePermission('unavailable', 'One-shot recognition is unavailable; Live or typed chat may still work.');
    setVoiceTray(true, 'failed', 'One-shot recognition unavailable', 'Use Live for cortex audio or send a typed message.');
    return;
  }
  const rec = new SpeechRecognition();
  rec.continuous = false;
  rec.interimResults = true;
  rec.lang = 'en-US';
  rec.onresult = event => {
    const entries = Array.from(event.results || []);
    const text = compact(entries.map(result => result?.[0]?.transcript || '').join(' '), 1_000);
    ui.voiceUserCaption.textContent = text || 'Listening…';
    const final = entries.length > 0 && entries.every(result => result.isFinal);
    if (final && text) onTranscript?.(text);
  };
  rec.onerror = event => {
    const denied = event.error === 'not-allowed' || event.error === 'service-not-allowed';
    setVoicePermission(denied ? 'denied' : 'unavailable');
    setVoiceTray(
      true,
      denied ? 'denied' : 'failed',
      denied ? 'Microphone blocked' : 'Recognition stopped',
      denied ? 'Enable microphone access in browser site settings.' : 'Use Live or typed chat to continue.',
    );
  };
  rec.onstart = () => {
    setVoicePermission('granted');
    ui.mic.setAttribute('aria-pressed', 'true');
    setVoiceTray(true, 'listening', 'Listening once', 'Finish your sentence and Weaver will receive the final caption.');
  };
  rec.onend = () => {
    ui.mic.setAttribute('aria-pressed', 'false');
    if (ui.voiceTray.dataset.state === 'listening') {
      setVoiceTray(true, 'idle', 'One-shot capture complete', 'The final caption was sent if speech was detected.');
    }
  };
  rec.start();
}

function openVoiceTray() {
  const stateName = state.realtime.live ? 'listening' : (state.realtime.connecting ? 'connecting' : 'idle');
  setVoiceTray(
    true,
    stateName,
    state.realtime.live ? 'Live cortex voice' : 'Voice session',
    state.realtime.lastStatus || 'Choose Live for continuous voice or Mic for one-shot recognition.',
  );
  void checkMicrophonePermission();
}

function collapseVoiceTray() {
  ui.voiceTray.dataset.open = 'false';
  ui.wake.focus({ preventScroll: true });
}

function realtimeVoiceAudit() {
  return {
    live: state.realtime.live,
    connecting: state.realtime.connecting,
    socketState: state.realtime.ws?.readyState ?? -1,
    model: state.realtime.model,
    voiceId: state.realtime.voiceId,
    mode: state.realtime.mode,
    inputRate: state.realtime.inputRate,
    outputRate: state.realtime.outputRate,
    bytesSent: state.realtime.bytesSent,
    bytesReceived: state.realtime.bytesReceived,
    framesSent: state.realtime.framesSent,
    framesReceived: state.realtime.framesReceived,
    droppedFrames: state.realtime.droppedFrames,
    reconnects: state.realtime.reconnects,
    runId: state.realtime.runId,
    lastStatus: state.realtime.lastStatus,
    lastError: state.realtime.lastError,
    lastHeard: state.realtime.lastHeard,
    lastSaid: state.realtime.lastSaid,
    lastRoute: state.realtime.lastRoute,
    cortexRouted: state.realtime.cortexRouted,
    lastReactionMs: state.realtime.lastReactionMs,
    lastSemanticLatencyMs: state.realtime.lastSemanticLatencyMs,
    reactionTargetMs: state.realtime.reactionTargetMs,
    lastAudioAt: state.realtime.lastAudioAt,
    playbackQueuedSeconds: audioCtx ? Math.max(0, state.realtime.playHead - audioCtx.currentTime) : 0,
    activeSources: state.realtime.activeSources.length,
    permission: state.realtime.permission,
    deviceAvailable: Boolean(state.realtime.deviceLabel),
    inputLevel: state.realtime.inputLevel,
    outputLevel: state.realtime.outputLevel,
    protocolVersion: state.realtime.protocolVersion,
    nextInputSequence: state.realtime.nextInputSequence,
    lastInputAckSequence: state.realtime.lastInputAckSequence,
    lastOutputSequence: state.realtime.lastOutputSequence,
    resumeTicketAvailable: Boolean(state.realtime.resumeToken),
    reconnectAttempt: state.realtime.reconnectAttempt,
    intentionalStop: state.realtime.intentionalStop,
    authenticatedTransport: 'HttpOnly session + WebSocket CSRF',
    nativeShellOwnsSensors: state.nativeShell,
  };
}

function audioAudit() {
  return {
    voiceReady: state.voiceReady,
    voiceMode: state.voiceMode,
    audioUnlocked: state.audioUnlocked,
    audioError: state.audioError,
    wakeSpoken: state.wakeSpoken,
    audioPaused: audio.paused,
    audioSrc: audio.currentSrc || audio.src || '',
    speaking: state.speaking,
    speechRun: state.speechRun,
    pendingVoice: Boolean(state.pendingVoiceUrl || state.pendingVoiceText),
    pendingVoiceUrl: Boolean(state.pendingVoiceUrl),
    pendingVoiceTextAvailable: Boolean(state.pendingVoiceText),
    lastVoiceTextAvailable: Boolean(state.lastVoiceText),
    lastVoiceError: compact(state.lastVoiceError, 80),
    lastTtsStatus: state.lastTtsStatus,
    lastTtsContentType: state.lastTtsContentType,
    lastPlaybackAt: state.lastPlaybackAt,
    audioContextState: audioCtx?.state || '',
    browserSpeech: canUseBrowserSpeech(),
    browserVoice: pickBrowserVoice()?.name || '',
    browserVoiceCount: browserVoices().length,
    realtime: realtimeVoiceAudit(),
  };
}

export {
  unlockAudio, replayPendingVoice, speak, startRealtimeVoice, stopRealtimeVoice,
  startMic, canUseBrowserSpeech, audioAudit, realtimeVoiceAudit,
  interruptVoice, replayLastVoice, openVoiceTray, collapseVoiceTray,
  checkMicrophonePermission, encodeVoiceFrame,
};
