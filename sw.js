// Service worker: the PWA foundation only (no push notifications in this stage).
// Network first, and "no-cache" so the browser's HTTP cache is always revalidated:
// caretakers always get the latest files. The cached copy is used only offline.
// Firebase (other origins) and the videos (range requests) are never cached here.
const CACHE = "gripsense-v2";

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
            const response = await fetch(request, { cache: "no-cache" });
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
