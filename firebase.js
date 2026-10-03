// Data layer: Firebase initialization and real-time listeners.
import { initializeApp } from "https://www.gstatic.com/firebasejs/12.3.0/firebase-app.js";
import {
    getDatabase,
    ref,
    query,
    orderByKey,
    limitToLast,
    onValue,
    set
} from "https://www.gstatic.com/firebasejs/12.3.0/firebase-database.js";
import {
    getAuth,
    onAuthStateChanged,
    signInWithEmailAndPassword,
    signOut
} from "https://www.gstatic.com/firebasejs/12.3.0/firebase-auth.js";
import { firebaseConfig } from "./firebase-config.js";

const HISTORY_LIMIT = 50;

const app = initializeApp(firebaseConfig);
const db = getDatabase(app);
const auth = getAuth(app);

// Calls onChange(user or null) on page load and whenever login state changes.
// Firebase keeps the session in the browser, so a caretaker stays logged in
// across visits until they log out.
export function watchAuth(onChange) {
    return onAuthStateChanged(auth, onChange);
}

export function login(email, password) {
    return signInWithEmailAndPassword(auth, email, password);
}

export function logout() {
    return signOut(auth);
}

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

export function savePatientName(name) {
    return set(ref(db, "patient/name"), name);
}

// Calls onChange(true/false) when the browser's connection to Firebase changes.
// This reflects the dashboard's connection only, not whether the ESP32 is online.
export function listenToConnection(onChange) {
    return onValue(ref(db, ".info/connected"), snapshot => onChange(snapshot.val() === true));
}
