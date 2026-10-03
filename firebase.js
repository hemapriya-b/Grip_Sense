// Data layer: Firebase initialization and real-time listeners.
import { initializeApp } from "https://www.gstatic.com/firebasejs/12.3.0/firebase-app.js";
import {
    getDatabase,
    ref,
    query,
    orderByKey,
    limitToLast,
    onValue
} from "https://www.gstatic.com/firebasejs/12.3.0/firebase-database.js";
import { firebaseConfig } from "./firebase-config.js";

const HISTORY_LIMIT = 50;

const app = initializeApp(firebaseConfig);
const db = getDatabase(app);

// Calls onData(rawEventsObject) whenever /events changes.
// Firebase push IDs are chronological, so ordering by key returns the most
// recently created events without needing a database index.
export function listenToEvents(onData, onError) {
    const eventsQuery = query(ref(db, "events"), orderByKey(), limitToLast(HISTORY_LIMIT));
    return onValue(eventsQuery, snapshot => onData(snapshot.val()), onError);
}

// Calls onData(nameOrNull) whenever /patient/name changes.
export function listenToPatientName(onData, onError) {
    return onValue(ref(db, "patient/name"), snapshot => onData(snapshot.val()), onError);
}

// Calls onChange(true/false) when the browser's connection to Firebase changes.
// This reflects the dashboard's connection only, not whether the ESP32 is online.
export function listenToConnection(onChange) {
    return onValue(ref(db, ".info/connected"), snapshot => onChange(snapshot.val() === true));
}
