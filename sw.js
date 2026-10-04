// Service worker: the PWA foundation only (no push notifications in this stage).
// Network first, so caretakers always get the latest app and data; the cached copy
// of the app files is used only when the network is unavailable.
// Firebase (other origins) and the videos (range requests) are never cached here.
const CACHE = "gripsense-v1";

self.addEventListener("install", () => self.skipWaiting());

self.addEventListener("activate", event => {
    event.waitUntil((async () => {
        for (const key of await caches.keys()) if (key !== CACHE) await caches.delete(key);
        await self.clients.claim();
    })());
});

self.addEventListener("fetch", event => {
    const { request } = event;
    const url = new URL(request.url);
    if (request.method !== "GET" || url.origin !== self.location.origin
        || url.pathname.includes("/videos/") || request.headers.has("range")) return;

    event.respondWith((async () => {
        try {
            const response = await fetch(request);
            if (response.ok) {
                const copy = response.clone();
                event.waitUntil(caches.open(CACHE).then(cache => cache.put(request, copy)));
            }
            return response;
        } catch (error) {
            const cached = await caches.match(request);
            if (cached) return cached;
            throw error;
        }
    })());
});
