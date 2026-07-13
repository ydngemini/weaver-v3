const CACHE_VERSION = 'weaver-headless-shell-v1';
const PUBLIC_SHELL = Object.freeze([
  '/',
  '/headless.html',
  '/manifest.webmanifest',
  '/weaver-logo.svg',
  '/headless/styles/tokens.css',
  '/headless/styles/shell.css',
  '/headless/js/core.js',
  '/headless/js/session.js',
  '/headless/js/voice-support.js',
  '/headless/js/visual-data.js',
  '/headless/js/visual-runtime.js',
  '/headless/js/voice.js',
  '/headless/js/visualization.js',
  '/headless/js/cortex.js',
  '/headless/js/state-channel.js',
  '/headless/js/lifecycle.js',
  '/headless/js/accessibility.js',
  '/headless/js/app.js',
  '/vendor/three.module.js',
]);
const PRIVATE_PREFIXES = Object.freeze(['/brain/', '/tts/', '/llm/', '/codebase/', '/gpu-render/']);

function isPrivateRequest(url) {
  return PRIVATE_PREFIXES.some(prefix => url.pathname.startsWith(prefix));
}

async function cachePublicResponse(cache, request, response) {
  if (response?.ok && response.type === 'basic' && !isPrivateRequest(new URL(request.url))) {
    await cache.put(request, response.clone());
  }
  return response;
}

async function networkFirstNavigation(request) {
  const cache = await caches.open(CACHE_VERSION);
  try {
    const response = await fetch(request);
    return cachePublicResponse(cache, request, response);
  } catch (error) {
    return (await cache.match(request, { ignoreSearch: true }))
      || (await cache.match('/headless.html'))
      || (await cache.match('/'));
  }
}

async function cacheFirstStatic(request) {
  const cache = await caches.open(CACHE_VERSION);
  const cached = await cache.match(request, { ignoreSearch: true });
  if (cached) return cached;
  const response = await fetch(request);
  return cachePublicResponse(cache, request, response);
}

self.addEventListener('install', event => {
  event.waitUntil(caches.open(CACHE_VERSION).then(cache => cache.addAll(PUBLIC_SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    const names = await caches.keys();
    await Promise.all(names.filter(name => name.startsWith('weaver-headless-shell-')
      && name !== CACHE_VERSION).map(name => caches.delete(name)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', event => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin || isPrivateRequest(url)) return;
  if (request.mode === 'navigate') {
    event.respondWith(networkFirstNavigation(request));
    return;
  }
  if (PUBLIC_SHELL.includes(url.pathname)) event.respondWith(cacheFirstStatic(request));
});
