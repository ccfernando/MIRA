const CACHE_NAME = "mira-shell-v3";
const APP_SHELL_URLS = [
  "/",
  "/upload",
  "/analytics",
  "/about",
  "/install",
  "/static/style.css",
  "/static/pwa.js?v=3",
  "/manifest.webmanifest",
  "/service-worker.js",
  "/static/icons/apple-touch-icon.png",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
  "/static/icons/icon-maskable-512.png",
  "/assets/img/mira.png",
  "/assets/img/mira-favicon-black.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL_URLS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") {
    return;
  }

  if (event.request.mode === "navigate") {
    event.respondWith(
      fetch(event.request).catch(() => caches.match(event.request).then((response) => response || caches.match("/")))
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then(
      (response) =>
        response ||
        fetch(event.request).then((networkResponse) => {
          const responseClone = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, responseClone));
          return networkResponse;
        })
    )
  );
});
