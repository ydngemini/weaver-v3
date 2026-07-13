// Browser voice UI, privacy-safe device metadata, and v2 binary framing.
import { ui, state, compact } from './core.js';

const VOICE_RECONNECT_POLICY = Object.freeze({
  initialDelayMs: 250,
  factor: 2,
  maxDelayMs: 8000,
  jitterRatio: 0.2,
  maxAttempts: 8,
});

function setVoiceTray(open, trayState = 'idle', title = '', detail = '') {
  ui.voiceTray.dataset.open = open ? 'true' : 'false';
  ui.voiceTray.dataset.state = trayState;
  if (title) ui.voiceSessionTitle.textContent = compact(title, 80);
  if (detail) ui.voiceSessionDetail.textContent = compact(detail, 180);
}

function setVoiceLevel(value) {
  const level = Math.max(0, Math.min(1, Number(value) || 0));
  state.realtime.inputLevel = level;
  ui.voiceMeterFill.style.transform = `scaleX(${Math.max(0.02, level)})`;
}

function setVoicePermission(permission, detail = '') {
  const value = ['prompt', 'granted', 'denied', 'unavailable', 'native'].includes(permission)
    ? permission
    : 'prompt';
  state.realtime.permission = value;
  state.realtime.permissionCheckedAt = Date.now();
  const labels = {
    prompt: 'Microphone: permission requested only after Live or Mic',
    granted: 'Microphone: allowed for this origin',
    denied: 'Microphone: blocked · enable it in browser site settings',
    unavailable: 'Microphone: unavailable on this device or browser',
    native: 'Microphone: owned by the authenticated native iOS shell',
  };
  ui.voicePermission.textContent = detail || labels[value];
}

function browserDeviceClass() {
  const iphone = /iPhone/i.test(navigator.userAgent);
  const iphone16eViewport = iphone && (
    (screen.width === 390 && screen.height === 844)
    || (screen.width === 844 && screen.height === 390)
  );
  return iphone16eViewport ? 'iphone-16e' : (iphone ? 'iphone' : 'web');
}

function safeDeviceTelemetry() {
  return {
    audioRoute: 'built-in',
    thermalState: 'unknown',
    lowPowerMode: state.lowPower,
    deviceClass: browserDeviceClass(),
  };
}

async function refreshVoiceDevice(stream = null) {
  let label = '';
  const track = stream?.getAudioTracks?.()[0];
  if (track?.label) label = compact(track.label, 80);
  if (!label && navigator.mediaDevices?.enumerateDevices && state.realtime.permission === 'granted') {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      label = compact(devices.find(device => device.kind === 'audioinput')?.label, 80);
    } catch (error) {}
  }
  state.realtime.deviceLabel = label;
  const deviceClass = browserDeviceClass();
  ui.voiceDevice.textContent = state.nativeShell
    ? 'Native iOS cortex bridge · AVFoundation owns capture'
    : `${label || 'Default microphone'} · ${deviceClass === 'web' ? 'browser fallback' : `${deviceClass} web fallback`}`;
}

async function checkMicrophonePermission() {
  if (state.nativeShell) {
    setVoicePermission('native');
    await refreshVoiceDevice();
    return 'native';
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    setVoicePermission('unavailable');
    return 'unavailable';
  }
  if (!navigator.permissions?.query) {
    setVoicePermission(state.realtime.permission === 'granted' ? 'granted' : 'prompt');
    return state.realtime.permission;
  }
  try {
    const status = await navigator.permissions.query({ name: 'microphone' });
    setVoicePermission(status.state);
    status.onchange = () => setVoicePermission(status.state);
    return status.state;
  } catch (error) {
    setVoicePermission(state.realtime.permission === 'granted' ? 'granted' : 'prompt');
    return state.realtime.permission;
  }
}

function realtimeVoiceUrl() {
  const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${scheme}//${location.host}/brain/realtime/voice`;
}

function writeUint64(view, offset, value) {
  const safe = Math.max(0, Number(value) || 0);
  const high = Math.floor(safe / 0x1_0000_0000);
  const low = safe >>> 0;
  view.setUint32(offset, high, false);
  view.setUint32(offset + 4, low, false);
}

function encodeVoiceFrame(pcm, sequence, capturedAtMs = Date.now()) {
  const audio = new Uint8Array(pcm);
  const frame = new Uint8Array(20 + audio.byteLength);
  frame.set([0x57, 0x56, 0x52, 0x32], 0);
  const view = new DataView(frame.buffer);
  writeUint64(view, 4, sequence);
  writeUint64(view, 12, capturedAtMs);
  frame.set(audio, 20);
  return frame.buffer;
}

function base64ToBytes(value) {
  const binary = atob(value || '');
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function downsampleToPcm16(input, inputRate, outputRate) {
  if (!input?.length || !inputRate || !outputRate) return new ArrayBuffer(0);
  const ratio = inputRate / outputRate;
  const length = Math.max(1, Math.floor(input.length / ratio));
  const pcm = new Int16Array(length);
  for (let index = 0; index < length; index++) {
    const start = Math.floor(index * ratio);
    const end = Math.min(input.length, Math.floor((index + 1) * ratio));
    let sum = 0;
    let count = 0;
    for (let sourceIndex = start; sourceIndex < end; sourceIndex++) {
      sum += input[sourceIndex];
      count += 1;
    }
    const sample = Math.max(-1, Math.min(1, count ? sum / count : input[start] || 0));
    pcm[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return pcm.buffer;
}

function reconnectDelay(attempt) {
  const base = Math.min(
    VOICE_RECONNECT_POLICY.maxDelayMs,
    VOICE_RECONNECT_POLICY.initialDelayMs * (VOICE_RECONNECT_POLICY.factor ** Math.max(0, attempt - 1)),
  );
  const jitter = base * VOICE_RECONNECT_POLICY.jitterRatio * ((Math.random() * 2) - 1);
  return Math.max(100, Math.round(base + jitter));
}

export {
  VOICE_RECONNECT_POLICY, setVoiceTray, setVoiceLevel, setVoicePermission,
  safeDeviceTelemetry, refreshVoiceDevice, checkMicrophonePermission,
  realtimeVoiceUrl, encodeVoiceFrame, base64ToBytes, downsampleToPcm16,
  reconnectDelay,
};
