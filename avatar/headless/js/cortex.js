import {
  ui, state, key, requestBrainKey, showCortexLocked, compact, dot, brainFetch,
  HEADLESS_MODEL_LABEL, setConnectionState,
} from './core.js';
import { ensureSession, sessionFetch } from './session.js';
import { speak } from './voice.js';

const MAX_SSE_BUFFER = 64 * 1024;
const MAX_PUBLIC_REPLY = 32 * 1024;
const MAX_HISTORY_MESSAGES = 20;
const MAX_HISTORY_CHARACTERS = 22_000;
const PUBLIC_EVENT_TYPES = new Set(['accepted', 'progress', 'delta', 'completed', 'cancelled', 'failed']);
const SPEAKER_EVENT_TYPES = new Set(['accepted', 'delta', 'completed', 'cancelled', 'failed']);
const TURN_ID_PATTERN = /^turn-[0-9a-f]{24}$/;
const PHASE_LABELS = Object.freeze({
  accepted: 'Weaver acknowledged you',
  queued: 'Weaver is preparing',
  thinking: 'Weaver is thinking',
  synthesizing: 'Weaver is forming her response',
});

let pollTimer = 0;
let pollFailures = 0;
const conversation = {
  history: [],
  active: null,
  lastAttempt: '',
  lastReply: '',
  lastOutcome: 'idle',
  sequence: 0,
};

function privateActivitySummary(kind, available, topics) {
  if (!available) return `No private ${kind} activity is available.`;
  const safeTopics = Array.isArray(topics)
    ? topics.filter(topic => typeof topic === 'string').slice(0, 4)
    : [];
  const topicText = safeTopics.length ? ` Safe topics: ${safeTopics.join(', ')}.` : '';
  return `Private ${kind} activity is available.${topicText} Content remains hidden.`;
}

function updateAwareness(snapshot) {
  const awareness = snapshot?.awareness && typeof snapshot.awareness === 'object'
    ? snapshot.awareness
    : {};
  const confidenceValue = Number(awareness.confidence);
  const confidence = Number.isFinite(confidenceValue)
    ? Math.max(0, Math.min(1, confidenceValue))
    : null;
  const awarenessStatus = ['nominal', 'limited', 'degraded', 'no-data'].includes(awareness.status)
    ? awareness.status
    : 'no-data';
  const reasons = Array.isArray(awareness.degraded_reasons)
    ? awareness.degraded_reasons.filter(reason => typeof reason === 'string').slice(0, 3)
    : [];
  ui.awarenessConfidence.textContent = confidence === null ? '—' : `${Math.round(confidence * 100)}%`;
  ui.awarenessConfidenceFill.style.width = confidence === null ? '0%' : `${Math.round(confidence * 100)}%`;
  ui.awarenessState.textContent = awarenessStatus;
  ui.awarenessReason.textContent = awarenessStatus === 'nominal'
    ? 'Body, world, cognition, fabric, and dependency signals are fused and nominal.'
    : (reasons.length ? `Bounded state: ${reasons.join(', ')}.` : 'Verified signals are incomplete or stale.');
  ui.diagnosticRevision.textContent = String(snapshot?.revision ?? '—');
  const generatedAt = Date.parse(snapshot?.generated_at || '');
  ui.diagnosticFreshness.textContent = Number.isFinite(generatedAt)
    ? `${Math.max(0, Math.round((Date.now() - generatedAt) / 1000))}s old`
    : 'unknown';
}

function applySnapshot(snapshot, { transport = 'polling fallback' } = {}) {
  if (snapshot?.schema_version !== 2 || !Number.isInteger(snapshot?.revision) || snapshot.revision < 1) {
    throw Object.assign(new Error('invalid-state-contract'), { code: 'invalid-state-contract' });
  }
  const priorRevision = Number(state.lastState?.revision || 0);
  if (snapshot.revision < priorRevision) return false;
  state.lastState = snapshot;
  const cognition = snapshot.cognition && typeof snapshot.cognition === 'object'
    ? snapshot.cognition
    : {};
  const system = snapshot.system && typeof snapshot.system === 'object'
    ? snapshot.system
    : {};
  ui.stamp.textContent = `${Math.max(0, Math.round(system.uptime_seconds || 0))}s active`;
  ui.thoughts.textContent = String(Math.max(0, Number(cognition.thought_count || 0)));
  ui.dreams.textContent = String(Math.max(0, Number(cognition.dream_count || 0)));
  ui.lastThought.textContent = privateActivitySummary(
    'thought', cognition.private_thought_available, cognition.thought_topics,
  );
  ui.lastDream.textContent = privateActivitySummary(
    'dream', cognition.private_dream_available, cognition.dream_topics,
  );
  const ready = Boolean(system.ready) && system.status !== 'inactive';
  dot(ui.brainDot, ready ? 'on' : 'busy');
  dot(ui.thoughtDot, cognition.private_thought_available ? 'on' : '');
  dot(ui.dreamDot, cognition.private_dream_available ? 'on' : '');
  ui.brain.textContent = ready ? HEADLESS_MODEL_LABEL : `cortex ${system.status || 'warming'}`;
  updateAwareness(snapshot);
  if (ui.diagnosticTransport) ui.diagnosticTransport.textContent = transport;
  const streamRecovering = ['connecting', 'reconnecting'].includes(state.channel.status)
    && !transport.includes('realtime');
  if (ready && streamRecovering) {
    setConnectionState('reconnecting', 'Realtime state is recovering. Authenticated polling remains active.');
  } else {
    setConnectionState(
      ready ? 'connected' : 'limited',
      ready ? 'Weaver cortex connected.' : 'Weaver cortex is available in a limited state.',
      { action: !ready },
    );
  }
  return true;
}

async function pollState() {
  if (!key() && state.auth.status !== 'ready') {
    showCortexLocked();
    setConnectionState('locked', 'Cortex is locked. Wake Weaver to connect.', { action: false });
    return false;
  }
  try {
    await ensureSession({ interactive: false });
    const response = await sessionFetch('/brain/headless/v2/state', { interactive: false });
    const snapshot = await response.json();
    const applied = applySnapshot(snapshot, { transport: state.channel.status === 'connected'
      ? 'realtime stream + polling safety' : 'authenticated polling fallback' });
    pollFailures = 0;
    state.connection.failures = 0;
    return applied;
  } catch (error) {
    pollFailures += 1;
    state.connection.failures = pollFailures;
    if (error?.code === 'WEAVER_CORTEX_LOCKED' || error?.code === 'authentication-required') {
      showCortexLocked();
      setConnectionState('locked', 'Weaver cortex is locked. Wake to reconnect.', { action: false });
    } else if (!navigator.onLine || error?.code === 'network-offline') {
      setConnectionState('offline', 'Offline shell active. Cortex actions will resume when the network returns.', { action: false });
    } else {
      ui.brain.textContent = 'cortex reconnecting';
      dot(ui.brainDot, 'busy');
      setConnectionState('reconnecting', 'Weaver cortex is recovering. Polling fallback remains active.');
    }
    return false;
  }
}

function defaultPollDelay() {
  if (document.hidden || !navigator.onLine) return 20_000;
  if (state.auth.status === 'locked' && !state.key) return 20_000;
  if (state.channel.status === 'connected') return 60_000;
  return Math.min(30_000, 2_000 * (2 ** Math.min(pollFailures, 4)));
}

function schedulePoll(delay = defaultPollDelay()) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async () => {
    await pollState();
    schedulePoll();
  }, delay);
}

async function checkModels() {
  return pollState();
}

function timestamp() {
  return new Intl.DateTimeFormat([], { hour: 'numeric', minute: '2-digit' }).format(new Date());
}

function removePlaceholders() {
  for (const placeholder of ui.transcript.querySelectorAll('[data-placeholder="true"]')) placeholder.remove();
}

function scrollTranscript() {
  requestAnimationFrame(() => {
    ui.transcript.scrollTop = ui.transcript.scrollHeight;
  });
}

async function copyText(value) {
  const text = String(value || '');
  if (!text) return false;
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (error) {
    const fallback = document.createElement('textarea');
    fallback.value = text;
    fallback.readOnly = true;
    fallback.className = 'sr-only';
    document.body.append(fallback);
    fallback.select();
    const copied = document.execCommand('copy');
    fallback.remove();
    return copied;
  }
}

function addMessageCopy(article, textProvider) {
  if (article.querySelector('.message-copy')) return;
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'message-copy';
  button.textContent = 'Copy';
  button.setAttribute('aria-label', 'Copy this message');
  button.addEventListener('click', async () => {
    const copied = await copyText(textProvider());
    button.textContent = copied ? 'Copied' : 'Copy failed';
    setTimeout(() => { button.textContent = 'Copy'; }, 1_200);
  });
  article.append(button);
}

function createMessage(role, text, { pending = false } = {}) {
  removePlaceholders();
  const isWeaver = role === 'assistant';
  const article = document.createElement('article');
  article.className = `message ${isWeaver ? 'message-weaver' : 'message-user'}`;
  article.dataset.role = isWeaver ? 'weaver' : 'user';
  article.dataset.state = pending ? 'pending' : 'complete';
  if (pending) article.setAttribute('aria-busy', 'true');

  const header = document.createElement('header');
  const label = document.createElement('span');
  label.textContent = isWeaver ? 'Weaver' : 'You';
  header.append(label);
  if (isWeaver) {
    const boundary = document.createElement('span');
    boundary.className = 'verified-speaker';
    boundary.textContent = 'verified boundary';
    header.append(boundary);
  } else {
    const time = document.createElement('time');
    time.textContent = timestamp();
    header.append(time);
  }

  const body = document.createElement('p');
  body.textContent = text;
  article.append(header, body);
  ui.transcript.insertBefore(article, ui.turnStatus);
  if (!pending) addMessageCopy(article, () => body.textContent);
  if (isWeaver) ui.said = body;
  else {
    ui.heard = body;
    ui.heardTime = header.querySelector('time');
  }
  scrollTranscript();
  return { article, body };
}

function setTurnUI(active, label = 'Weaver is thinking') {
  const running = Boolean(active);
  ui.turnStatus.hidden = !running;
  ui.stopTurn.hidden = !running;
  ui.stopTurn.disabled = !running;
  ui.send.disabled = running;
  ui.retryTurn.disabled = running || !conversation.lastAttempt;
  ui.copyLast.disabled = running || !conversation.lastReply;
  ui.transcript.setAttribute('aria-busy', running ? 'true' : 'false');
  if (running) {
    ui.turnStatusText.textContent = label;
    ui.turnElapsed.textContent = '0s';
    dot(ui.brainDot, 'busy');
  } else {
    dot(ui.brainDot, state.lastState?.system?.ready ? 'on' : '');
  }
}

function updateElapsed(active, elapsedMs = performance.now() - active.startedAt) {
  const seconds = Math.max(0, Math.floor(Number(elapsedMs || 0) / 1_000));
  ui.turnElapsed.textContent = `${seconds}s`;
}

function boundedHistory(nextMessage) {
  const available = MAX_HISTORY_CHARACTERS - nextMessage.length;
  const selected = [];
  let characters = 0;
  for (let index = conversation.history.length - 1; index >= 0; index--) {
    const item = conversation.history[index];
    if (selected.length >= MAX_HISTORY_MESSAGES || characters + item.content.length > available) break;
    selected.unshift({ role: item.role, content: item.content });
    characters += item.content.length;
  }
  return selected;
}

function trimConversationHistory() {
  while (conversation.history.length > MAX_HISTORY_MESSAGES) conversation.history.shift();
  let characters = conversation.history.reduce((total, item) => total + item.content.length, 0);
  while (characters > MAX_HISTORY_CHARACTERS && conversation.history.length) {
    characters -= conversation.history.shift().content.length;
  }
}

function extractSSEFrames(buffer, flush = false) {
  const frames = [];
  let remaining = buffer;
  let match = /\r?\n\r?\n/.exec(remaining);
  while (match) {
    frames.push(remaining.slice(0, match.index));
    remaining = remaining.slice(match.index + match[0].length);
    match = /\r?\n\r?\n/.exec(remaining);
  }
  if (flush && remaining.trim()) {
    frames.push(remaining);
    remaining = '';
  }
  return { frames, remaining };
}

function parseSSEFrame(frame) {
  const data = frame
    .split(/\r?\n/)
    .filter(line => line.startsWith('data:'))
    .map(line => line.slice(5).trimStart())
    .join('\n');
  if (!data) return null;
  if (data.length > 16_384) throw Object.assign(new Error('invalid-stream-contract'), { code: 'invalid-stream-contract' });
  const event = JSON.parse(data);
  if (!event || typeof event !== 'object' || Array.isArray(event)) {
    throw Object.assign(new Error('invalid-stream-contract'), { code: 'invalid-stream-contract' });
  }
  return event;
}

function requireWeaverBoundary(event) {
  if (!PUBLIC_EVENT_TYPES.has(event.type)) {
    throw Object.assign(new Error('invalid-stream-contract'), { code: 'invalid-stream-contract' });
  }
  if (SPEAKER_EVENT_TYPES.has(event.type) && event.speaker !== 'weaver') {
    throw Object.assign(new Error('speaker-boundary-rejected'), { code: 'speaker-boundary-rejected' });
  }
  if (event.turnId !== undefined && !TURN_ID_PATTERN.test(String(event.turnId))) {
    throw Object.assign(new Error('invalid-stream-contract'), { code: 'invalid-stream-contract' });
  }
}

function handleTurnEvent(active, event) {
  requireWeaverBoundary(event);
  if (active.serverTurnId && event.turnId && event.turnId !== active.serverTurnId) {
    throw Object.assign(new Error('invalid-stream-contract'), { code: 'invalid-stream-contract' });
  }
  if (event.type === 'accepted') {
    active.serverTurnId = event.turnId;
    active.accepted = true;
    const reaction = Number(event.reactionMs);
    const target = Number(event.reactionTargetMs || 200);
    if (Number.isFinite(reaction) && Number.isFinite(target)) {
      ui.reactionReadout.textContent = `Reaction ${Math.round(reaction)} ms · target ≤${Math.round(target)} ms`;
      ui.diagnosticReaction.textContent = `${Math.round(reaction)} ms ack / ≤${Math.round(target)} ms target`;
    }
    ui.turnStatusText.textContent = PHASE_LABELS.accepted;
    state.pulse = 0.9;
  } else if (event.type === 'progress') {
    const label = PHASE_LABELS[event.phase];
    if (!label) throw Object.assign(new Error('invalid-stream-contract'), { code: 'invalid-stream-contract' });
    ui.turnStatusText.textContent = label;
    if (Number.isFinite(Number(event.elapsedMs))) updateElapsed(active, Number(event.elapsedMs));
  } else if (event.type === 'delta') {
    const index = Number(event.index);
    if (!Number.isInteger(index) || index < 0) {
      throw Object.assign(new Error('invalid-stream-contract'), { code: 'invalid-stream-contract' });
    }
    if (index < active.nextChunk) return;
    if (index !== active.nextChunk || typeof event.text !== 'string') {
      throw Object.assign(new Error('invalid-stream-contract'), { code: 'invalid-stream-contract' });
    }
    if (active.reply.length + event.text.length > MAX_PUBLIC_REPLY) {
      throw Object.assign(new Error('response-too-large'), { code: 'response-too-large' });
    }
    active.nextChunk += 1;
    active.reply += event.text;
    active.message.body.textContent = active.reply;
    scrollTranscript();
  } else if (event.type === 'completed') {
    active.completed = true;
    updateElapsed(active, Number(event.elapsedMs));
  } else if (event.type === 'cancelled') {
    active.cancelled = true;
  } else if (event.type === 'failed') {
    const error = new Error('service-unavailable');
    error.code = /^[a-z][a-z0-9-]{1,63}$/.test(String(event.code || ''))
      ? event.code
      : 'service-unavailable';
    error.retryable = Boolean(event.retryable);
    throw error;
  }
}

async function consumeSSE(response, active) {
  if (!response.body?.getReader) {
    throw Object.assign(new Error('stream-unavailable'), { code: 'stream-unavailable' });
  }
  const contentType = response.headers.get('content-type') || '';
  if (!contentType.toLowerCase().startsWith('text/event-stream')) {
    throw Object.assign(new Error('invalid-stream-contract'), { code: 'invalid-stream-contract' });
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      if (buffer.length > MAX_SSE_BUFFER) {
        throw Object.assign(new Error('stream-buffer-exceeded'), { code: 'stream-buffer-exceeded' });
      }
      const extracted = extractSSEFrames(buffer, done);
      buffer = extracted.remaining;
      for (const frame of extracted.frames) {
        const event = parseSSEFrame(frame);
        if (event) handleTurnEvent(active, event);
      }
      if (done) break;
    }
  } finally {
    reader.releaseLock();
  }
}

function safeFailureMessage(error) {
  if (error?.code === 'WEAVER_CORTEX_LOCKED' || error?.code === 'authentication-required') {
    return 'Cortex session unavailable. Wake Weaver to reconnect, then retry.';
  }
  if (error?.code === 'rate-limited') return 'Weaver is at capacity. Your message is safe to retry shortly.';
  if (error?.code === 'speaker-boundary-rejected') return 'A response was blocked because it did not pass Weaver’s speaker boundary.';
  if (error?.code === 'invalid-request') return 'That message could not be accepted. Edit it and retry.';
  return 'Weaver could not complete this turn. The connection can remain open while you retry.';
}

function completeMessage(active) {
  const reply = active.reply.trim();
  if (!reply) throw Object.assign(new Error('empty-public-response'), { code: 'empty-public-response' });
  active.message.body.textContent = reply;
  active.message.article.dataset.state = 'complete';
  active.message.article.removeAttribute('aria-busy');
  addMessageCopy(active.message.article, () => active.message.body.textContent);
  conversation.history.push(
    { role: 'user', content: active.text },
    { role: 'assistant', content: reply },
  );
  trimConversationHistory();
  conversation.lastReply = reply;
  conversation.lastOutcome = 'completed';
  state.pulse = 1;
  void speak(reply);
  void pollState();
}

function cancelMessage(active) {
  active.message.article.dataset.state = 'cancelled';
  active.message.article.classList.add('message-cancelled');
  active.message.article.removeAttribute('aria-busy');
  active.message.body.textContent = active.reply.trim() || 'Turn stopped before Weaver produced a public response.';
  conversation.lastOutcome = 'cancelled';
  ui.turnStatusText.textContent = 'Turn stopped';
}

function failMessage(active, error) {
  active.message.article.dataset.state = 'failed';
  active.message.article.classList.add('message-error');
  active.message.article.removeAttribute('aria-busy');
  active.message.body.textContent = safeFailureMessage(error);
  conversation.lastOutcome = 'failed';
  setConnectionState('limited', 'The last Weaver turn failed and can be retried.');
}

async function chat(value) {
  const text = String(value || '').trim().slice(0, 4_000);
  if (!text || conversation.active) return false;
  conversation.lastAttempt = text;
  const history = boundedHistory(text);
  const userMessage = createMessage('user', text);
  const weaverMessage = createMessage('assistant', 'Acknowledging…', { pending: true });
  const active = {
    id: ++conversation.sequence,
    clientTurnId: `client-${Date.now().toString(36)}-${conversation.sequence.toString(36)}`,
    serverTurnId: '',
    text,
    userMessage,
    message: weaverMessage,
    controller: new AbortController(),
    startedAt: performance.now(),
    elapsedTimer: 0,
    nextChunk: 0,
    reply: '',
    accepted: false,
    completed: false,
    cancelled: false,
    cancelRequested: false,
  };
  conversation.active = active;
  conversation.lastOutcome = 'pending';
  setTurnUI(active, 'Opening a secure Weaver turn');
  active.elapsedTimer = setInterval(() => updateElapsed(active), 1_000);
  ui.heard.textContent = text;
  scrollTranscript();

  try {
    const response = await sessionFetch('/brain/headless/v2/chat/stream', {
      method: 'POST',
      interactive: true,
      signal: active.controller.signal,
      headers: { Accept: 'text/event-stream' },
      body: JSON.stringify({
        message: text,
        history,
        client_turn_id: active.clientTurnId,
        max_tokens: 512,
      }),
    });
    await consumeSSE(response, active);
    if (active.cancelled || active.cancelRequested) cancelMessage(active);
    else if (active.completed) completeMessage(active);
    else throw Object.assign(new Error('incomplete-stream'), { code: 'incomplete-stream' });
    return active.completed && !active.cancelled;
  } catch (error) {
    if (active.cancelRequested || error?.name === 'AbortError') cancelMessage(active);
    else failMessage(active, error);
    return false;
  } finally {
    clearInterval(active.elapsedTimer);
    if (conversation.active === active) conversation.active = null;
    setTurnUI(null);
    scrollTranscript();
  }
}

async function cancelActiveTurn({ remote = true } = {}) {
  const active = conversation.active;
  if (!active || active.cancelRequested) return false;
  active.cancelRequested = true;
  ui.turnStatusText.textContent = 'Stopping turn';
  ui.stopTurn.disabled = true;
  if (remote && active.serverTurnId) {
    try {
      await sessionFetch(`/brain/headless/v2/chat/${active.serverTurnId}`, {
        method: 'DELETE',
        interactive: false,
      });
    } catch (error) {}
  }
  active.controller.abort();
  return true;
}

async function retryLastTurn() {
  if (!conversation.lastAttempt || conversation.active) return false;
  return chat(conversation.lastAttempt);
}

async function copyLastReply() {
  const copied = await copyText(conversation.lastReply);
  if (copied) {
    ui.copyLast.textContent = 'Copied';
    setTimeout(() => { ui.copyLast.textContent = 'Copy'; }, 1_200);
  }
  return copied;
}

async function triggerDream() {
  if (!requestBrainKey()) {
    ui.lastDream.textContent = 'cortex locked · Wake to connect';
    showCortexLocked();
    return;
  }
  dot(ui.dreamDot, 'busy');
  try {
    await brainFetch('/brain/trigger/dream', {
      method: 'POST',
      body: JSON.stringify({ reason: 'operator-headless-button' }),
    });
    ui.lastDream.textContent = 'Private dream activity completed. Content remains hidden.';
    state.pulse = 1;
    dot(ui.dreamDot, 'on');
  } catch (error) {
    ui.lastDream.textContent = 'Dream cycle unavailable. Private content remains hidden.';
    dot(ui.dreamDot, '');
  }
}

function conversationAudit() {
  return {
    historyMessages: conversation.history.length,
    active: Boolean(conversation.active),
    activeAccepted: Boolean(conversation.active?.accepted),
    activePhase: conversation.active ? ui.turnStatusText.textContent : 'idle',
    lastOutcome: conversation.lastOutcome,
    lastReplyAvailable: Boolean(conversation.lastReply),
    maxSemanticWaitMs: null,
    speakerBoundary: 'weaver-only',
  };
}

export {
  pollState, schedulePoll, checkModels, chat, triggerDream,
  cancelActiveTurn, retryLastTurn, copyLastReply, conversationAudit,
  extractSSEFrames, parseSSEFrame, applySnapshot,
};
