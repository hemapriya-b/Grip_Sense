// App entry: connect Firebase data to application state and the UI.
import {
    watchAuth,
    login,
    logout,
    listenToEvents,
    listenToPatientName,
    listenToConnection
} from "./firebase.js";
import { parseEvents } from "./events.js";
import { render, showView, setUser, onLoginSubmit, onLogoutClick, setLoginBusy, setLoginError } from "./ui.js";

// If no data has arrived after this long, show the error state instead of
// an endless loading message (Firebase does not report "offline" as an error).
const LOAD_TIMEOUT_MS = 10000;

const initialState = () => ({
    events: [],
    eventsLoaded: false,
    patientName: null,
    patientNameLoaded: false,
    connected: false,
    error: null
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
