// Viewport, connectivity, install, and public-shell service-worker lifecycle.
import { state, ui, dot, setConnectionState } from './core.js';

let deferredInstallPrompt = null;
let viewportRaf = 0;
let resizeCallback = () => {};

function currentDisplayMode() {
  if (navigator.standalone === true) return 'standalone';
  if (matchMedia('(display-mode: standalone)').matches) return 'standalone';
  if (matchMedia('(display-mode: fullscreen)').matches) return 'fullscreen';
  return 'browser';
}

function updateNetwork(announce = true) {
  state.lifecycle.online = navigator.onLine;
  if (ui.networkStatus) ui.networkStatus.textContent = navigator.onLine ? 'online' : 'offline shell';
  if (ui.networkPill) ui.networkPill.dataset.state = navigator.onLine ? 'online' : 'offline';
  if (ui.networkDot) dot(ui.networkDot, navigator.onLine ? 'on' : 'busy');
  if (announce && ui.connectionAnnouncement) {
    if (navigator.onLine) {
      setConnectionState('reconnecting', 'Network restored. Weaver is reconnecting.');
      dispatchEvent(new CustomEvent('weaver:network-restored'));
    } else {
      setConnectionState('offline', 'Network lost. The public shell remains available; cortex actions are paused.', { action: false });
      dispatchEvent(new CustomEvent('weaver:network-lost'));
    }
  }
}

function updateViewport() {
  viewportRaf = 0;
  const viewport = globalThis.visualViewport;
  const visualWidth = Math.max(1, Math.round(viewport?.width || innerWidth));
  const visualHeight = Math.max(1, Math.round(viewport?.height || innerHeight));
  const offsetTop = Math.max(0, Math.round(viewport?.offsetTop || 0));
  const keyboardInset = Math.max(0, Math.round(innerHeight - visualHeight - offsetTop));
  const orientation = innerWidth >= innerHeight ? 'landscape' : 'portrait';
  state.render.orientation = orientation;
  state.render.viewportWidth = Math.max(1, innerWidth);
  state.render.viewportHeight = Math.max(1, innerHeight);
  state.render.visualViewportWidth = visualWidth;
  state.render.visualViewportHeight = visualHeight;
  state.render.keyboardInset = keyboardInset;
  document.documentElement.style.setProperty('--app-viewport-height', `${visualHeight}px`);
  document.documentElement.style.setProperty('--keyboard-inset', `${keyboardInset}px`);
  document.body.dataset.orientation = orientation;
  resizeCallback();
}

function queueViewportUpdate() {
  if (viewportRaf) return;
  viewportRaf = requestAnimationFrame(updateViewport);
}

function updateInstallUI() {
  state.lifecycle.installAvailable = Boolean(deferredInstallPrompt);
  state.lifecycle.displayMode = currentDisplayMode();
  state.lifecycle.installed = state.lifecycle.displayMode !== 'browser';
  if (ui.installApp) {
    ui.installApp.hidden = !deferredInstallPrompt || state.lifecycle.installed || state.nativeShell;
  }
}

async function requestInstall() {
  const promptEvent = deferredInstallPrompt;
  if (!promptEvent) return false;
  deferredInstallPrompt = null;
  updateInstallUI();
  await promptEvent.prompt();
  const choice = await promptEvent.userChoice;
  state.lifecycle.installed = choice?.outcome === 'accepted';
  updateInstallUI();
  return state.lifecycle.installed;
}

async function registerPublicShell() {
  if (!('serviceWorker' in navigator) || state.nativeShell) {
    state.lifecycle.serviceWorker = state.nativeShell ? 'native-render-only' : 'unsupported';
    return null;
  }
  if (!isSecureContext && !['localhost', '127.0.0.1'].includes(location.hostname)) {
    state.lifecycle.serviceWorker = 'insecure-context';
    return null;
  }
  state.lifecycle.serviceWorker = 'registering';
  try {
    const registration = await navigator.serviceWorker.register('/headless-sw.js', {
      scope: '/',
      updateViaCache: 'none',
    });
    state.lifecycle.serviceWorker = 'ready';
    state.lifecycle.controlled = Boolean(navigator.serviceWorker.controller);
    registration.addEventListener('updatefound', () => {
      state.lifecycle.serviceWorker = 'updating';
      const worker = registration.installing;
      worker?.addEventListener('statechange', () => {
        if (worker.state === 'activated') state.lifecycle.serviceWorker = 'ready';
      });
    });
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      state.lifecycle.controlled = true;
      state.lifecycle.serviceWorker = 'ready';
    });
    return registration;
  } catch (error) {
    state.lifecycle.serviceWorker = 'failed';
    return null;
  }
}

function lifecycleAudit() {
  return {
    online: state.lifecycle.online,
    serviceWorker: state.lifecycle.serviceWorker,
    controlled: state.lifecycle.controlled,
    installAvailable: state.lifecycle.installAvailable,
    installed: state.lifecycle.installed,
    displayMode: state.lifecycle.displayMode,
    nativeOwnership: state.nativeShell ? 'swiftui-cortex-bridge' : 'web-fallback',
    orientation: state.render.orientation,
    viewport: {
      layoutWidth: state.render.viewportWidth,
      layoutHeight: state.render.viewportHeight,
      visualWidth: state.render.visualViewportWidth,
      visualHeight: state.render.visualViewportHeight,
      keyboardInset: state.render.keyboardInset,
    },
  };
}

function initLifecycle({ onViewportChange = () => {} } = {}) {
  resizeCallback = onViewportChange;
  updateNetwork(false);
  updateViewport();
  updateInstallUI();
  addEventListener('online', () => updateNetwork(true), { passive: true });
  addEventListener('offline', () => updateNetwork(true), { passive: true });
  addEventListener('resize', queueViewportUpdate, { passive: true });
  addEventListener('orientationchange', queueViewportUpdate, { passive: true });
  globalThis.screen?.orientation?.addEventListener?.('change', queueViewportUpdate);
  globalThis.visualViewport?.addEventListener('resize', queueViewportUpdate, { passive: true });
  globalThis.visualViewport?.addEventListener('scroll', queueViewportUpdate, { passive: true });
  addEventListener('beforeinstallprompt', event => {
    event.preventDefault();
    deferredInstallPrompt = event;
    updateInstallUI();
  });
  addEventListener('appinstalled', () => {
    deferredInstallPrompt = null;
    state.lifecycle.installed = true;
    updateInstallUI();
  });
  ui.installApp?.addEventListener('click', () => { void requestInstall(); });
  void registerPublicShell();
  return lifecycleAudit;
}

export { initLifecycle, lifecycleAudit, queueViewportUpdate, requestInstall };
