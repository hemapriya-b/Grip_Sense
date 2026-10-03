// App entry: connect Firebase data to application state and the UI.
import {
    watchAuth,
    login,
    logout,
    listenToEvents,
    listenToPatientName,
    listenToConnection,
    savePatientName,
    acknowledgeEvent
} from "./firebase.js";
import { parseEvents } from "./events.js";
import {
    render,
    showView,
    setUser,
    onLoginSubmit,
    onLogoutClick,
    setLoginBusy,
    setLoginError,
    onPatientSave,
    setPatientBusy,
    setPatientFeedback,
    onAcknowledgeClick
} from "./ui.js";

// If no data has arrived after this long, show the error state instead of
// an endless loading message (Firebase does not report "offline" as an error).
const LOAD_TIMEOUT_MS = 10000;
const PATIENT_NAME_MAX_LENGTH = 50;

const initialState = () => ({
    events: [],
    eventsLoaded: false,
    patientName: null,
    patientNameLoaded: false,
    connected: false,
    error: null,
    acknowledging: new Set(),   // event IDs with a save in progress
    ackError: null
});

let state = initialState();
let stopListeners = [];
let loadTimer = null;

function update(changes) {
    Object.assign(state, changes);
    render(state);
}

function handleError(error) {
    console.error("GripSense: Firebase error", error);
    update({ error });
}

function startDataListeners() {
    state = initialState();
    render(state);

    loadTimer = setTimeout(() => {
        if (!state.eventsLoaded) handleError(new Error("Timed out waiting for /events"));
    }, LOAD_TIMEOUT_MS);

    stopListeners = [
        listenToEvents(raw => {
            clearTimeout(loadTimer);
            update({ events: parseEvents(raw), eventsLoaded: true, error: null });
        }, handleError),

        listenToPatientName(name => {
            const trimmed = typeof name === "string" ? name.trim() : "";
            update({ patientName: trimmed || null, patientNameLoaded: true });
        }, handleError),

        listenToConnection(connected => update({ connected }))
    ];
}

function stopDataListeners() {
    clearTimeout(loadTimer);
    stopListeners.forEach(stop => stop());
    stopListeners = [];
    state = initialState();
}

// Friendly text for Firebase Auth error codes.
function loginErrorMessage(error) {
    switch (error.code) {
        case "auth/invalid-credential":
        case "auth/invalid-login-credentials":
        case "auth/wrong-password":
        case "auth/user-not-found":
            return "Incorrect email or password.";
        case "auth/invalid-email":
            return "Please enter a valid email address.";
        case "auth/user-disabled":
            return "This caretaker account has been disabled.";
        case "auth/too-many-requests":
            return "Too many failed attempts. Please wait a few minutes and try again.";
        case "auth/network-request-failed":
            return "Unable to reach the login service. Please check your internet connection.";
        case "auth/operation-not-allowed":
        case "auth/configuration-not-found":
            return "Email/Password login is not enabled for this project.";
        default:
            return `Login failed (${error.code || error.message}).`;
    }
}

onLoginSubmit(async (email, password) => {
    setLoginError("");
    setLoginBusy(true);
    try {
        await login(email, password);
        // watchAuth below switches to the dashboard.
    } catch (error) {
        console.error("GripSense: login failed", error);
        setLoginError(loginErrorMessage(error));
    } finally {
        setLoginBusy(false);
    }
});

// Returns an error message, or null when the name is acceptable.
function validatePatientName(name) {
    if (!name) return "Please enter the patient name.";
    if (name.length > PATIENT_NAME_MAX_LENGTH) {
        return `Patient name must be ${PATIENT_NAME_MAX_LENGTH} characters or fewer.`;
    }
    if (/\p{Cc}/u.test(name)) return "Patient name contains invalid characters.";
    return null;
}

onPatientSave(async rawName => {
    const name = rawName.trim();
    const problem = validatePatientName(name);
    if (problem) {
        setPatientFeedback(problem, "error");
        return;
    }
    // Firebase would otherwise queue the write silently until reconnecting.
    if (!state.connected) {
        setPatientFeedback("Not connected. Please check your internet connection and try again.", "error");
        return;
    }

    setPatientFeedback("");
    setPatientBusy(true);
    try {
        await savePatientName(name);
        setPatientFeedback(`Patient name saved: ${name}`, "success");
    } catch (error) {
        console.error("GripSense: saving patient name failed", error);
        setPatientFeedback(`Could not save the patient name (${error.code || error.message}).`, "error");
    } finally {
        setPatientBusy(false);
    }
});

onAcknowledgeClick(async eventId => {
    const event = state.events.find(e => e.id === eventId);
    // Acknowledgement is final: never re-write an already acknowledged event.
    if (!event || event.acknowledged || state.acknowledging.has(eventId)) return;
    if (!state.connected) {
        update({ ackError: "Not connected. The alert was not acknowledged. Please check your internet connection." });
        return;
    }

    state.acknowledging.add(eventId);
    update({ ackError: null });
    try {
        await acknowledgeEvent(eventId);
    } catch (error) {
        console.error("GripSense: acknowledge failed", error);
        update({ ackError: `Could not acknowledge the alert (${error.code || error.message}).` });
    } finally {
        state.acknowledging.delete(eventId);
        render(state);
    }
});

onLogoutClick(() => {
    logout().catch(error => console.error("GripSense: logout failed", error));
});

showView("checking");

watchAuth(user => {
    stopDataListeners();
    setUser(user);
    if (user) {
        showView("dashboard");
        startDataListeners();
    } else {
        showView("login");
    }
});
