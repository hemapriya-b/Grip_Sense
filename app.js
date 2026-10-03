// App entry: connect Firebase data to application state and the UI.
import { listenToEvents, listenToPatientName, listenToConnection } from "./firebase.js";
import { parseEvents } from "./events.js";
import { render } from "./ui.js";

// If no data has arrived after this long, show the error state instead of
// an endless loading message (Firebase does not report "offline" as an error).
const LOAD_TIMEOUT_MS = 10000;

const state = {
    events: [],
    eventsLoaded: false,
    patientName: null,
    patientNameLoaded: false,
    connected: false,
    error: null
};

function update(changes) {
    Object.assign(state, changes);
    render(state);
}

function handleError(error) {
    console.error("GripSense: Firebase error", error);
    update({ error });
}

render(state);

const loadTimer = setTimeout(() => {
    if (!state.eventsLoaded) handleError(new Error("Timed out waiting for /events"));
}, LOAD_TIMEOUT_MS);

listenToEvents(raw => {
    clearTimeout(loadTimer);
    update({ events: parseEvents(raw), eventsLoaded: true, error: null });
}, handleError);

listenToPatientName(name => {
    const trimmed = typeof name === "string" ? name.trim() : "";
    update({ patientName: trimmed || null, patientNameLoaded: true });
}, handleError);

listenToConnection(connected => update({ connected }));
