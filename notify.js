// Alert output: browser-generated sound patterns and system notifications.
// Browsers decide whether either is allowed; nothing here bypasses those rules.
import { ACTIONS } from "./events.js";

// Rhythms follow the ESP32 buzzer patterns (fast/slow gaps per need), with
// shorter tones so a browser tab is not harsh. Each plays for at most 30 s.
const SOUND_PATTERNS = {
    [ACTIONS.LIGHT_PRESS]:    { frequency: 660, onMs: 1000, offMs: 1500 },   // ESP32: 10 s on / 5 s off
    [ACTIONS.HARD_PRESS]:     { frequency: 880, onMs: 1000, offMs: 400 },    // ESP32: 10 s on / 2 s off
    [ACTIONS.DOUBLE_SQUEEZE]: { frequency: 523, onMs: 300,  offMs: 2000 }    // ESP32: 0.3 s on / 2 s off
};
const SOUND_DURATION_MS = 30000;
const VOLUME = 0.25;

let audioContext = null;
let playing = null;   // { eventId, stop() }

function getAudioContext() {
    if (!audioContext) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) return null;
        audioContext = new AudioContextClass();
    }
    return audioContext;
}

// "running" | "locked" (needs a user click first) | "unsupported"
export function audioStatus() {
    if (!window.AudioContext && !window.webkitAudioContext) return "unsupported";
    return audioContext?.state === "running" ? "running" : "locked";
}

// Must be called from a user gesture (click/tap) to satisfy autoplay rules.
export async function unlockAudio() {
    const context = getAudioContext();
    if (!context) return "unsupported";
    if (context.state !== "running") {
        try { await context.resume(); } catch { /* still locked */ }
    }
    return audioStatus();
}

// Plays the pattern for one event; replaces any pattern already playing.
// Returns false if the browser has not allowed audio yet.
export function playAlertSound(eventId, action) {
    stopAlertSound();
    const pattern = SOUND_PATTERNS[action];
    const context = audioContext;
    if (!pattern || !context || context.state !== "running") return false;

    const gain = context.createGain();
    gain.gain.value = 0;
    gain.connect(context.destination);
    const oscillator = context.createOscillator();
    oscillator.type = "sine";
    oscillator.frequency.value = pattern.frequency;
    oscillator.connect(gain);

    // Schedule every on/off cycle up front, with short ramps to avoid clicks.
    const start = context.currentTime + 0.05;
    const end = start + SOUND_DURATION_MS / 1000;
    const on = pattern.onMs / 1000;
    const cycle = (pattern.onMs + pattern.offMs) / 1000;
    for (let t = start; t < end; t += cycle) {
        const stopAt = Math.min(t + on, end);
        gain.gain.setValueAtTime(0, t);
        gain.gain.linearRampToValueAtTime(VOLUME, t + 0.02);
        gain.gain.setValueAtTime(VOLUME, Math.max(t + 0.02, stopAt - 0.02));
        gain.gain.linearRampToValueAtTime(0, stopAt);
    }
    oscillator.start(start);
    oscillator.stop(end);

    const current = {
        eventId,
        stop() {
            try { oscillator.stop(); } catch { /* already stopped */ }
            oscillator.disconnect();
            gain.disconnect();
        }
    };
    oscillator.onended = () => { if (playing === current) playing = null; };
    playing = current;
    return true;
}

export function stopAlertSound() {
    playing?.stop();
    playing = null;
}

export function playingEventId() {
    return playing?.eventId ?? null;
}

// ---- System notifications (shown only when the caretaker is not looking at the app) ----

// "granted" | "denied" | "default" (not asked yet) | "unsupported"
export function notificationStatus() {
    return "Notification" in window ? Notification.permission : "unsupported";
}

// Must be called from a user gesture; browsers ignore or block unprompted requests.
export async function requestNotificationPermission() {
    if (!("Notification" in window)) return "unsupported";
    try { return await Notification.requestPermission(); } catch { return Notification.permission; }
}

export function isAppInView() {
    return document.visibilityState === "visible" && document.hasFocus();
}

// Returns true if a notification was shown. The browser/OS decides the sound.
export function showSystemNotification(eventId, text) {
    if (notificationStatus() !== "granted") return false;
    try {
        const notification = new Notification("GripSense", {
            body: text,
            tag: eventId,               // one notification per event
            requireInteraction: true,   // stays until the caretaker reacts, where supported
            silent: false
        });
        notification.onclick = () => {
            window.focus();
            notification.close();
        };
        return true;
    } catch (error) {
        // Some mobile browsers only allow notifications from a service worker.
        console.warn("GripSense: notification not shown", error);
        return false;
    }
}
