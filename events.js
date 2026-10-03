// Event logic: convert Firebase data into validated, sorted event objects.

// Technical action names sent by the ESP32. This is a fixed contract.
export const ACTIONS = {
    LIGHT_PRESS: "LIGHT_PRESS",
    HARD_PRESS: "HARD_PRESS",
    DOUBLE_SQUEEZE: "DOUBLE_SQUEEZE"
};

// The meaning layer. All patient-facing wording lives here.
export const NEED_MAPPINGS = {
    [ACTIONS.LIGHT_PRESS]: {
        icon: "🟢",
        label: "Hungry",
        color: "green",
        message: name => `${name} is hungry.`
    },
    [ACTIONS.HARD_PRESS]: {
        icon: "🔴",
        label: "Pain",
        color: "red",
        message: name => `${name} is in pain.`
    },
    [ACTIONS.DOUBLE_SQUEEZE]: {
        icon: "🔵",
        label: "Washroom",
        color: "blue",
        message: name => `${name} needs to use the washroom.`
    }
};

function isValidEvent(raw) {
    return raw !== null
        && typeof raw === "object"
        && Object.hasOwn(NEED_MAPPINGS, raw.action)
        && Number.isFinite(raw.timestamp)
        && Number.isFinite(raw.maxPressure);
}

// Firebase object { id: {...}, ... } → array of events, newest first.
// Invalid records are skipped and reported in the console, never shown.
export function parseEvents(rawEvents) {
    if (!rawEvents) return [];

    const events = [];
    for (const [id, raw] of Object.entries(rawEvents)) {
        if (!isValidEvent(raw)) {
            console.warn("GripSense: skipping invalid event", id, raw);
            continue;
        }
        events.push({
            id,
            action: raw.action,
            maxPressure: raw.maxPressure,
            timestamp: raw.timestamp,
            acknowledged: raw.acknowledged === true,
            acknowledgedAt: Number.isFinite(raw.acknowledgedAt) ? raw.acknowledgedAt : null
        });
    }

    events.sort((a, b) => b.timestamp - a.timestamp);
    return events;
}

// Picks the event for the main alert card from a newest-first list:
// the newest unacknowledged event, or the newest event if all are acknowledged.
export function selectLatestAlert(events) {
    const unacknowledged = events.filter(event => !event.acknowledged);
    return {
        event: unacknowledged[0] ?? events[0] ?? null,
        allAcknowledged: unacknowledged.length === 0,
        moreUnacknowledged: Math.max(unacknowledged.length - 1, 0)
    };
}
