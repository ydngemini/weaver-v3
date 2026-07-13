// Short-lived browser-session boundary. The long-lived key is sent only while
// exchanging it for an HttpOnly cookie; ordinary requests use cookie + CSRF.
import { state, key, requestBrainKey, clearBrainKey } from './core.js';

const SESSION_PATH = '/brain/headless/v2/session';
const SESSION_RENEW_PATH = '/brain/headless/v2/session/renew';
const SAFE_TOKEN = /^[A-Za-z0-9_-]{32,64}$/;
let sessionPromise = null;

function correlationId(prefix = 'web') {
  const bytes = new Uint8Array(12);
  crypto.getRandomValues(bytes);
  const suffix = Array.from(bytes, value => value.toString(16).padStart(2, '0')).join('');
  return `${prefix}-${suffix}`;
}

function invalidateSession(reason = 'authentication-required') {
  clearTimeout(state.auth.renewTimer);
  state.auth.renewTimer = 0;
  state.auth.status = 'locked';
  state.auth.csrfToken = '';
  state.auth.expiresAt = 0;
  state.auth.lastError = reason;
  state.auth.generation += 1;
}

function sessionError(code, status = 0, retryable = false) {
  const error = new Error(code);
  error.code = code;
  error.status = status;
  error.retryable = Boolean(retryable);
  return error;
}

async function responseError(response) {
  let code = response.status === 403 ? 'authentication-required' : 'service-unavailable';
  let retryable = response.status >= 500 || response.status === 429;
  try {
    const payload = await response.json();
    const publicError = payload?.error;
    if (typeof publicError?.code === 'string' && /^[a-z][a-z0-9-]{1,63}$/.test(publicError.code)) {
      code = publicError.code;
      retryable = Boolean(publicError.retryable);
    }
  } catch (error) {}
  return sessionError(code, response.status, retryable);
}

function acceptSession(payload) {
  const expiresAt = Date.parse(payload?.expires_at || '');
  const expiresIn = Number(payload?.expires_in_seconds);
  const remainingMs = expiresAt - Date.now();
  if (
    payload?.schema_version !== 2
    || !SAFE_TOKEN.test(String(payload?.csrf_token || ''))
    || !Number.isFinite(expiresAt)
    || !Number.isFinite(expiresIn)
    || expiresIn < 1
    || expiresIn > 3600
    || remainingMs < 500
    || remainingMs > 3_605_000
    || Math.abs(remainingMs - expiresIn * 1_000) > 15_000
  ) {
    throw sessionError('invalid-session-contract');
  }
  state.auth.status = 'ready';
  state.auth.csrfToken = payload.csrf_token;
  state.auth.expiresAt = expiresAt;
  state.auth.lastError = '';
  state.auth.generation += 1;
  scheduleRenewal(expiresIn);
  return true;
}

function scheduleRenewal(expiresInSeconds) {
  clearTimeout(state.auth.renewTimer);
  const earlySeconds = Math.min(60, Math.max(10, expiresInSeconds * 0.2));
  const delay = Math.max(5_000, (expiresInSeconds - earlySeconds) * 1_000);
  state.auth.renewTimer = setTimeout(() => {
    renewSession().catch(() => invalidateSession('session-renewal-failed'));
  }, delay);
}

async function bootstrapSession(interactive) {
  if (!navigator.onLine) throw sessionError('network-offline', 0, true);
  const brainKey = key() || (interactive ? requestBrainKey() : '');
  if (!brainKey) throw sessionError('WEAVER_CORTEX_LOCKED');
  state.auth.status = 'connecting';
  const response = await fetch(SESSION_PATH, {
    method: 'POST',
    credentials: 'same-origin',
    cache: 'no-store',
    headers: {
      'X-Weaver-Key': brainKey,
      'X-Correlation-ID': correlationId('session'),
    },
  });
  if (!response.ok) {
    const error = await responseError(response);
    invalidateSession(error.code);
    if (response.status === 403) clearBrainKey();
    throw error;
  }
  return acceptSession(await response.json());
}

async function renewSession() {
  if (!navigator.onLine) throw sessionError('network-offline', 0, true);
  if (!state.auth.csrfToken) throw sessionError('authentication-required', 403);
  state.auth.status = 'renewing';
  const response = await fetch(SESSION_RENEW_PATH, {
    method: 'POST',
    credentials: 'same-origin',
    cache: 'no-store',
    headers: {
      'X-Weaver-CSRF': state.auth.csrfToken,
      'X-Correlation-ID': correlationId('renew'),
    },
  });
  if (!response.ok) {
    const error = await responseError(response);
    invalidateSession(error.code);
    throw error;
  }
  return acceptSession(await response.json());
}

async function ensureSession({ interactive = true } = {}) {
  const remaining = state.auth.expiresAt - Date.now();
  if (state.auth.status === 'ready' && state.auth.csrfToken && remaining > 30_000) return true;
  if (state.auth.csrfToken && remaining > 0) {
    try { return await renewSession(); } catch (error) {}
  }
  if (!sessionPromise) {
    sessionPromise = bootstrapSession(interactive).finally(() => { sessionPromise = null; });
  }
  return sessionPromise;
}

async function sessionFetch(path, options = {}) {
  await ensureSession({ interactive: options.interactive !== false });
  const method = String(options.method || 'GET').toUpperCase();
  const hasBody = options.body !== undefined && options.body !== null;
  const requestHeaders = {
    'X-Correlation-ID': correlationId('request'),
    ...(hasBody ? { 'Content-Type': 'application/json' } : {}),
    ...options.headers,
  };
  if (!['GET', 'HEAD', 'OPTIONS'].includes(method)) {
    requestHeaders['X-Weaver-CSRF'] = state.auth.csrfToken;
  }
  const response = await fetch(path, {
    ...options,
    method,
    credentials: 'same-origin',
    cache: 'no-store',
    headers: requestHeaders,
  });
  if (!response.ok) {
    const error = await responseError(response);
    if (response.status === 401 || response.status === 403) invalidateSession(error.code);
    throw error;
  }
  return response;
}

function authAudit() {
  return {
    status: state.auth.status,
    authenticated: Boolean(state.auth.csrfToken && state.auth.expiresAt > Date.now()),
    expiresInMs: Math.max(0, state.auth.expiresAt - Date.now()),
    generation: state.auth.generation,
    storedLongLivedKey: false,
    transport: 'HttpOnly cookie + rotating CSRF',
    lastError: state.auth.lastError,
  };
}

export {
  ensureSession, renewSession, invalidateSession, sessionFetch, authAudit,
};
