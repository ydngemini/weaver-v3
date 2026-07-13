// Keyboard, sensory preference, focus containment, and redacted operator view.
import { state, ui, configurePerformance } from './core.js';

const PREFERENCE_KEY = 'weaver_accessibility_v1';
const FOCUSABLE = [
  'button:not([disabled]):not([hidden])',
  'a[href]',
  'input:not([disabled])',
  'textarea:not([disabled])',
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

let priorFocus = null;
let diagnosticsTimer = 0;
let visualPreferenceCallback = () => {};

function readPreferences() {
  try {
    const parsed = JSON.parse(localStorage.getItem(PREFERENCE_KEY) || '{}');
    return {
      reduceMotion: parsed.reduceMotion === true,
      highContrast: parsed.highContrast === true,
      hideField: parsed.hideField === true,
    };
  } catch (error) {
    return { reduceMotion: false, highContrast: false, hideField: false };
  }
}

function savePreferences() {
  try {
    localStorage.setItem(PREFERENCE_KEY, JSON.stringify({
      reduceMotion: state.userReducedMotion,
      highContrast: state.userHighContrast,
      hideField: state.fieldHidden,
    }));
  } catch (error) {}
}

function updatePreferenceUi() {
  const effectiveContrast = state.systemHighContrast || state.userHighContrast;
  if (ui.motionToggle) {
    ui.motionToggle.setAttribute('aria-pressed', state.userReducedMotion ? 'true' : 'false');
    ui.motionToggle.textContent = state.userReducedMotion ? 'On' : 'Off';
  }
  if (ui.motionStatus) {
    ui.motionStatus.textContent = state.systemReducedMotion
      ? 'Reduced by the operating system'
      : state.userReducedMotion ? 'Reduced motion enabled here' : 'Following system preference';
  }
  if (ui.contrastToggle) {
    ui.contrastToggle.setAttribute('aria-pressed', state.userHighContrast ? 'true' : 'false');
    ui.contrastToggle.textContent = state.userHighContrast ? 'On' : 'Off';
  }
  if (ui.contrastStatus) {
    ui.contrastStatus.textContent = state.systemHighContrast
      ? 'Higher contrast requested by the system'
      : effectiveContrast ? 'Higher contrast enabled here' : 'Following system preference';
  }
  if (ui.fieldToggle) {
    ui.fieldToggle.setAttribute('aria-pressed', state.fieldHidden ? 'true' : 'false');
    ui.fieldToggle.textContent = state.fieldHidden ? 'Hidden' : 'Visible';
  }
  if (ui.fieldStatus) ui.fieldStatus.textContent = state.fieldHidden
    ? 'Ambient field hidden; status remains available as text'
    : 'Semantic field visible';
}

function applyPreferences({ persist = false } = {}) {
  state.reducedMotion = state.systemReducedMotion || state.userReducedMotion;
  document.body.dataset.reduceMotion = state.reducedMotion ? 'true' : 'false';
  document.body.dataset.highContrast = state.systemHighContrast || state.userHighContrast ? 'true' : 'false';
  document.body.dataset.fieldHidden = state.fieldHidden ? 'true' : 'false';
  configurePerformance();
  state.lastFrame = 0;
  updatePreferenceUi();
  if (persist) savePreferences();
  visualPreferenceCallback();
}

function resetPreferences() {
  state.userReducedMotion = false;
  state.userHighContrast = false;
  state.fieldHidden = false;
  applyPreferences({ persist: true });
}

function visibleFocusableElements() {
  return [...ui.diagnosticsDrawer.querySelectorAll(FOCUSABLE)].filter(element => (
    !element.hidden && element.getAttribute('aria-hidden') !== 'true'
    && (element.offsetWidth > 0 || element.offsetHeight > 0)
  ));
}

function refreshOperatorDiagnostics() {
  if (ui.diagnosticConnection) {
    const messageAge = state.channel.lastMessageAt
      ? `${Math.max(0, Math.round((Date.now() - state.channel.lastMessageAt) / 1_000))}s stream age`
      : 'stream waiting';
    ui.diagnosticConnection.textContent = `${state.connection.status} · ${messageAge} · ${state.connection.failures} poll failures`;
  }
  if (ui.diagnosticTransport) {
    ui.diagnosticTransport.textContent = state.channel.status === 'connected'
      ? 'authenticated realtime state'
      : `polling fallback · stream ${state.channel.status}`;
  }
  if (ui.diagnosticSession) {
    const seconds = Math.max(0, Math.round((state.auth.expiresAt - Date.now()) / 1_000));
    ui.diagnosticSession.textContent = state.auth.status === 'ready'
      ? `ready · ${seconds}s remaining` : state.auth.status;
  }
  if (ui.diagnosticFreshness) {
    const generatedAt = Date.parse(state.lastState?.generated_at || '');
    ui.diagnosticFreshness.textContent = Number.isFinite(generatedAt)
      ? `${Math.max(0, Math.round((Date.now() - generatedAt) / 1_000))}s old`
      : 'waiting';
  }
  if (ui.diagnosticNetwork) {
    ui.diagnosticNetwork.textContent = `${state.lifecycle.online ? 'online' : 'offline'} · shell ${state.lifecycle.serviceWorker}`;
  }
  if (ui.diagnosticRender) {
    ui.diagnosticRender.textContent = `${state.render.profile} · ${state.targetFps} fps · ${state.render.effectiveDpr.toFixed(2)} DPR`;
  }
  if (ui.diagnosticVoice) {
    const voiceStatus = state.realtime.live ? 'live' : state.realtime.connecting ? 'connecting' : 'idle';
    ui.diagnosticVoice.textContent = `${voiceStatus} · ${state.realtime.reconnects} reconnects`;
  }
  if (ui.diagnosticPrivacy) ui.diagnosticPrivacy.textContent = 'private content hidden · Weaver-only output';
}

function setDiagnosticsOpen(open) {
  const next = Boolean(open);
  if (next === (ui.diagnosticsDrawer.dataset.open === 'true')) return;
  if (next) priorFocus = document.activeElement;
  ui.diagnosticsDrawer.dataset.open = next ? 'true' : 'false';
  ui.diagnosticsDrawer.setAttribute('aria-hidden', next ? 'false' : 'true');
  ui.diagnosticsDrawer.inert = !next;
  ui.appShell.inert = next;
  ui.diagnosticsToggle.setAttribute('aria-expanded', next ? 'true' : 'false');
  ui.diagnosticsScrim.hidden = !next;
  clearInterval(diagnosticsTimer);
  diagnosticsTimer = 0;
  if (next) {
    refreshOperatorDiagnostics();
    diagnosticsTimer = setInterval(refreshOperatorDiagnostics, 1_000);
    setTimeout(() => ui.diagnosticsClose.focus({ preventScroll: true }), 0);
  } else {
    const restore = priorFocus?.isConnected ? priorFocus : ui.diagnosticsToggle;
    priorFocus = null;
    setTimeout(() => restore.focus({ preventScroll: true }), 0);
  }
}

function handleDiagnosticsKeydown(event) {
  if (ui.diagnosticsDrawer.dataset.open !== 'true') return false;
  if (event.key === 'Escape') {
    event.preventDefault();
    setDiagnosticsOpen(false);
    return true;
  }
  if (event.key !== 'Tab') return false;
  const focusable = visibleFocusableElements();
  if (!focusable.length) {
    event.preventDefault();
    ui.diagnosticsDrawer.focus({ preventScroll: true });
    return true;
  }
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && (document.activeElement === first || document.activeElement === ui.diagnosticsDrawer)) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
  return true;
}

function accessibilityAudit() {
  const coarsePointer = matchMedia('(pointer: coarse)').matches;
  const visibleButtons = [...document.querySelectorAll('button:not([hidden])')]
    .filter(button => button.offsetWidth > 0 && button.offsetHeight > 0);
  const smallestTarget = visibleButtons.reduce(
    (minimum, button) => Math.min(minimum, button.getBoundingClientRect().height),
    Infinity,
  );
  return {
    language: document.documentElement.lang,
    systemReducedMotion: state.systemReducedMotion,
    userReducedMotion: state.userReducedMotion,
    effectiveReducedMotion: state.reducedMotion,
    systemHighContrast: state.systemHighContrast,
    userHighContrast: state.userHighContrast,
    fieldHidden: state.fieldHidden,
    preferencesStoredWithoutCredentials: true,
    drawerOpen: ui.diagnosticsDrawer.dataset.open === 'true',
    backgroundInert: ui.appShell.inert,
    focusInsideDrawer: ui.diagnosticsDrawer.contains(document.activeElement),
    coarsePointer,
    smallestVisibleTargetPx: Number.isFinite(smallestTarget) ? Math.round(smallestTarget) : null,
    liveRegions: document.querySelectorAll('[aria-live]').length,
    privateCognitionInDiagnostics: false,
  };
}

function initAccessibility({ onVisualPreference = () => {} } = {}) {
  visualPreferenceCallback = onVisualPreference;
  const saved = readPreferences();
  state.userReducedMotion = saved.reduceMotion;
  state.userHighContrast = saved.highContrast;
  state.fieldHidden = saved.hideField;
  applyPreferences();

  const motionQuery = matchMedia('(prefers-reduced-motion: reduce)');
  motionQuery.addEventListener?.('change', event => {
    state.systemReducedMotion = event.matches;
    applyPreferences();
  });
  const contrastQuery = matchMedia('(prefers-contrast: more)');
  contrastQuery.addEventListener?.('change', event => {
    state.systemHighContrast = event.matches;
    applyPreferences();
  });

  ui.motionToggle?.addEventListener('click', () => {
    state.userReducedMotion = !state.userReducedMotion;
    applyPreferences({ persist: true });
  });
  ui.contrastToggle?.addEventListener('click', () => {
    state.userHighContrast = !state.userHighContrast;
    applyPreferences({ persist: true });
  });
  ui.fieldToggle?.addEventListener('click', () => {
    state.fieldHidden = !state.fieldHidden;
    applyPreferences({ persist: true });
  });
  ui.resetPreferences?.addEventListener('click', resetPreferences);
  refreshOperatorDiagnostics();
}

export {
  initAccessibility, setDiagnosticsOpen, handleDiagnosticsKeydown,
  refreshOperatorDiagnostics, accessibilityAudit, resetPreferences,
};
