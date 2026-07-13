// Authenticated, read-mostly public state stream with polling fallback.
import { state, ui, setConnectionState } from './core.js';
import { ensureSession } from './session.js';
import { applySnapshot, pollState, schedulePoll } from './cortex.js';

const PUBLIC_FIELDS = Object.freeze(['freshness', 'system', 'awareness', 'voice', 'cognition', 'fabric']);
const SNAPSHOT_KEYS = new Set(['schema_version', 'revision', 'generated_at', ...PUBLIC_FIELDS]);
const DELTA_KEYS = new Set(['schema_version', 'base_revision', 'revision', 'generated_at', 'changes']);
const MAX_MESSAGE_CHARACTERS = 65_536;
const MAX_RECONNECT_ATTEMPTS = 8;

function exactKeys(value, expected) {
  const keys = Object.keys(value || {});
  return keys.length === expected.size && keys.every(key => expected.has(key));
}

function validTimestamp(value) {
  return typeof value === 'string' && Number.isFinite(Date.parse(value));
}

function validSnapshot(snapshot) {
  return Boolean(
    snapshot && typeof snapshot === 'object' && !Array.isArray(snapshot)
    && exactKeys(snapshot, SNAPSHOT_KEYS)
    && snapshot.schema_version === 2
    && Number.isInteger(snapshot.revision) && snapshot.revision >= 1
    && validTimestamp(snapshot.generated_at)
    && PUBLIC_FIELDS.every(field => snapshot[field] && typeof snapshot[field] === 'object'
      && !Array.isArray(snapshot[field])),
  );
}

function validDelta(delta) {
  if (!delta || typeof delta !== 'object' || Array.isArray(delta) || !exactKeys(delta, DELTA_KEYS)) return false;
  const changes = delta.changes;
  const changeKeys = changes && typeof changes === 'object' && !Array.isArray(changes)
    ? Object.keys(changes) : [];
  return delta.schema_version === 2
    && Number.isInteger(delta.base_revision) && delta.base_revision >= 0
    && Number.isInteger(delta.revision) && delta.revision > delta.base_revision
    && validTimestamp(delta.generated_at)
    && changeKeys.length > 0
    && changeKeys.every(field => PUBLIC_FIELDS.includes(field)
      && changes[field] && typeof changes[field] === 'object' && !Array.isArray(changes[field]));
}

function parseStateMessage(raw) {
  if (typeof raw !== 'string' || raw.length > MAX_MESSAGE_CHARACTERS) throw new Error('invalid-state-message');
  const message = JSON.parse(raw);
  if (!message || typeof message !== 'object' || Array.isArray(message) || typeof message.type !== 'string') {
    throw new Error('invalid-state-message');
  }
  if (message.type === 'hello') {
    if (!exactKeys(message, new Set(['type', 'schema_version', 'correlation_id', 'heartbeat_interval_ms', 'revision']))
      || message.schema_version !== 2
      || !Number.isInteger(message.revision) || message.revision < 0
      || !Number.isInteger(message.heartbeat_interval_ms)
      || message.heartbeat_interval_ms < 1_000 || message.heartbeat_interval_ms > 10_000
      || !/^[A-Za-z0-9_.:-]{1,64}$/.test(String(message.correlation_id || ''))) {
      throw new Error('invalid-state-message');
    }
  } else if (message.type === 'snapshot') {
    if (!exactKeys(message, new Set(['type', 'snapshot'])) || !validSnapshot(message.snapshot)) {
      throw new Error('invalid-state-message');
    }
  } else if (message.type === 'delta') {
    if (!exactKeys(message, new Set(['type', 'delta'])) || !validDelta(message.delta)) {
      throw new Error('invalid-state-message');
    }
  } else if (message.type === 'heartbeat') {
    if (!exactKeys(message, new Set(['type', 'sent_at', 'revision']))
      || !validTimestamp(message.sent_at) || !Number.isInteger(message.revision) || message.revision < 0) {
      throw new Error('invalid-state-message');
    }
  } else if (message.type === 'progress') {
    if (!exactKeys(message, new Set(['type', 'turn_id', 'phase', 'elapsed_ms']))
      || !/^[A-Za-z0-9_.:-]{1,64}$/.test(String(message.turn_id || ''))
      || !['accepted', 'queued', 'thinking', 'synthesizing', 'completed', 'cancelled', 'failed'].includes(message.phase)
      || !Number.isInteger(message.elapsed_ms) || message.elapsed_ms < 0 || message.elapsed_ms > 180_000) {
      throw new Error('invalid-state-message');
    }
  } else if (message.type === 'capsule_receipt') {
    const decision = ['allow', 'revise', 'block', 'not-evaluated'];
    if (!exactKeys(message, new Set(['type', 'capsule_id', 'status', 'decision', 'correlation_id']))
      || !/^cap-[0-9a-f]{24}$/.test(String(message.capsule_id || ''))
      || !['verified', 'evaluated', 'rejected'].includes(message.status)
      || !decision.includes(message.decision)
      || !/^[A-Za-z0-9_.:-]{1,64}$/.test(String(message.correlation_id || ''))) {
      throw new Error('invalid-state-message');
    }
  } else if (message.type === 'error') {
    if (!exactKeys(message, new Set(['type', 'code', 'retryable', 'correlation_id']))
      || !/^[a-z][a-z0-9-]{1,63}$/.test(String(message.code || ''))
      || typeof message.retryable !== 'boolean'
      || !/^[A-Za-z0-9_.:-]{1,64}$/.test(String(message.correlation_id || ''))) {
      throw new Error('invalid-state-message');
    }
  } else {
    throw new Error('invalid-state-message');
  }
  return message;
}

function stateChannelUrl() {
  const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${scheme}//${location.host}/brain/headless/v2/stream`;
}

function clearChannelTimers() {
  clearTimeout(state.channel.reconnectTimer);
  clearTimeout(state.channel.watchdogTimer);
  state.channel.reconnectTimer = 0;
  state.channel.watchdogTimer = 0;
}

function armWatchdog() {
  clearTimeout(state.channel.watchdogTimer);
  const timeout = Math.max(15_000, state.channel.heartbeatIntervalMs * 3 + 5_000);
  state.channel.watchdogTimer = setTimeout(() => {
    state.channel.lastError = 'state-heartbeat-missed';
    state.channel.ws?.close(4000, 'heartbeat missed');
  }, timeout);
}

function sendResume(ws) {
  if (ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: 'resume', revision: Math.max(0, Number(state.lastState?.revision || 0)) }));
}

function applyDelta(delta) {
  const current = state.lastState;
  const currentRevision = Number(current?.revision || 0);
  if (delta.revision <= currentRevision) return true;
  if (!current || delta.base_revision !== currentRevision) return false;
  const snapshot = {
    ...current,
    ...delta.changes,
    schema_version: 2,
    revision: delta.revision,
    generated_at: delta.generated_at,
  };
  if (!validSnapshot(snapshot)) return false;
  applySnapshot(snapshot, { transport: 'authenticated realtime state' });
  state.channel.lastRevision = snapshot.revision;
  return true;
}

function markConnected() {
  state.channel.status = 'connected';
  state.channel.attempt = 0;
  state.channel.lastError = '';
  if (ui.diagnosticTransport) ui.diagnosticTransport.textContent = 'authenticated realtime state';
  setConnectionState('connected', 'Realtime Weaver state connected.');
  schedulePoll(60_000);
}

function reconnectDelay(attempt) {
  const base = Math.min(8_000, 250 * (2 ** Math.max(0, attempt - 1)));
  return Math.max(150, Math.round(base * (0.85 + Math.random() * 0.30)));
}

function scheduleStateReconnect(reason = 'state stream interrupted') {
  if (state.channel.intentionalStop || !navigator.onLine || document.hidden) return;
  state.channel.attempt += 1;
  state.channel.reconnects += 1;
  state.channel.lastError = reason;
  schedulePoll(0);
  if (state.channel.attempt > MAX_RECONNECT_ATTEMPTS) {
    state.channel.status = 'polling';
    setConnectionState('failed', 'Realtime state is unavailable. Authenticated polling remains active.');
    return;
  }
  state.channel.status = 'reconnecting';
  setConnectionState(
    'reconnecting',
    `Realtime state is reconnecting (${state.channel.attempt}/${MAX_RECONNECT_ATTEMPTS}). Polling remains active.`,
  );
  clearTimeout(state.channel.reconnectTimer);
  state.channel.reconnectTimer = setTimeout(() => { void startStateChannel(); }, reconnectDelay(state.channel.attempt));
}

async function startStateChannel({ interactive = false, force = false } = {}) {
  if (state.nativeShell || !navigator.onLine || document.hidden) return false;
  const active = state.channel.ws;
  if (!force && active && [WebSocket.CONNECTING, WebSocket.OPEN].includes(active.readyState)) return true;
  if (force && active) active.close(1000, 'operator reconnect');
  clearTimeout(state.channel.reconnectTimer);
  state.channel.reconnectTimer = 0;
  state.channel.intentionalStop = false;
  state.channel.status = 'connecting';
  try {
    await ensureSession({ interactive });
  } catch (error) {
    state.channel.status = navigator.onLine ? 'polling' : 'offline';
    state.channel.lastError = error?.code || 'authentication-required';
    return false;
  }
  if (!state.auth.csrfToken) return false;
  const ws = new WebSocket(stateChannelUrl(), [
    'weaver-headless-v2',
    `weaver-csrf.${state.auth.csrfToken}`,
  ]);
  state.channel.ws = ws;
  ws.addEventListener('open', () => {
    if (state.channel.ws !== ws) return;
    state.channel.lastMessageAt = Date.now();
    armWatchdog();
  });
  ws.addEventListener('message', event => {
    if (state.channel.ws !== ws) return;
    try {
      const message = parseStateMessage(event.data);
      state.channel.lastMessageAt = Date.now();
      armWatchdog();
      if (message.type === 'hello') {
        state.channel.heartbeatIntervalMs = message.heartbeat_interval_ms;
        sendResume(ws);
      } else if (message.type === 'snapshot') {
        applySnapshot(message.snapshot, { transport: 'authenticated realtime state' });
        state.channel.lastRevision = message.snapshot.revision;
        markConnected();
      } else if (message.type === 'delta') {
        if (!applyDelta(message.delta)) sendResume(ws);
        else markConnected();
      } else if (message.type === 'heartbeat') {
        state.channel.lastHeartbeatAt = Date.now();
        if (message.revision > Number(state.lastState?.revision || 0)) sendResume(ws);
        if (state.channel.status !== 'connected') markConnected();
      } else if (message.type === 'capsule_receipt') {
        state.channel.capsuleReceipts += 1;
      } else if (message.type === 'error' && message.code === 'state-resync-required') {
        void pollState().then(() => sendResume(ws));
      }
    } catch (error) {
      state.channel.lastError = 'invalid-state-message';
      ws.close(1008, 'invalid state contract');
    }
  });
  ws.addEventListener('error', () => {
    state.channel.lastError = 'state-transport-error';
  });
  ws.addEventListener('close', event => {
    if (state.channel.ws !== ws) return;
    state.channel.ws = null;
    clearTimeout(state.channel.watchdogTimer);
    state.channel.watchdogTimer = 0;
    if (state.channel.intentionalStop) return;
    scheduleStateReconnect(event.code === 1008 ? 'state authentication expired' : 'state stream interrupted');
  });
  return true;
}

function stopStateChannel({ permanent = true } = {}) {
  state.channel.intentionalStop = permanent;
  clearChannelTimers();
  const ws = state.channel.ws;
  state.channel.ws = null;
  if (ws && [WebSocket.CONNECTING, WebSocket.OPEN].includes(ws.readyState)) ws.close(1000, 'client pause');
  state.channel.status = permanent ? 'stopped' : 'paused';
}

function stateChannelAudit() {
  return {
    status: state.channel.status,
    authenticated: state.channel.status === 'connected' && state.auth.status === 'ready',
    lastRevision: state.channel.lastRevision,
    heartbeatAgeMs: state.channel.lastHeartbeatAt ? Date.now() - state.channel.lastHeartbeatAt : null,
    messageAgeMs: state.channel.lastMessageAt ? Date.now() - state.channel.lastMessageAt : null,
    heartbeatIntervalMs: state.channel.heartbeatIntervalMs,
    reconnectAttempt: state.channel.attempt,
    reconnects: state.channel.reconnects,
    pollingFallback: state.channel.status !== 'connected',
    capsuleReceipts: state.channel.capsuleReceipts,
    canExecuteCapsules: false,
    maxSemanticWaitMs: null,
    lastError: state.channel.lastError,
  };
}

export {
  startStateChannel, stopStateChannel, stateChannelAudit,
  parseStateMessage, validSnapshot, validDelta, applyDelta,
};
