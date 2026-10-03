// App entry: route the signed-in caretaker, and connect Firebase data to the UI.
import {
    watchAuth,
    login,
    logout,
    signup,
    getCaretaker,
    getPatient,
    saveCaretaker,
    savePatientProfile,
    listenToEvents,
    listenToPatientName,
    listenToConnection,
    savePatientName,
    acknowledgeEvent
} from "./firebase.js";
import { parseEvents } from "./events.js";
import {
    validatePatientName,
    validateUsername,
    validateSignup,
    validatePatientProfile,
    isPatientComplete
} from "./profile.js";
import {
    render,
    showView,
    setChecking,
    onRetryClick,
    setUser,
    onFormSubmit,
    setFieldErrors,
    setFormError,
    setFormBusy,
    fillPatientSetup,
    onLogoutClick,
    onPatientSave,
    setPatientBusy,
    setPatientFeedback,
    onAcknowledgeClick
} from "./ui.js";

// If no data has arrived after this long, show the error state instead of
// an endless loading message (Firebase does not report "offline" as an error).
const LOAD_TIMEOUT_MS = 10000;

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

let currentUser = null;
let signupInProgress = false;   // routing is done by the signup handler itself
let routeId = 0;                // ignores stale async results after logout/login

function update(changes) {
    Object.assign(state, changes);
    render(state);
}

function handleError(error) {
    console.error("GripSense: Firebase error", error);
    update({ error });
}

function withTimeout(promise) {
    return Promise.race([
        promise,
        new Promise((_, reject) => setTimeout(() => reject(new Error("Timed out")), LOAD_TIMEOUT_MS))
    ]);
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

// ---- Routing after authentication ----
// caretaker profile missing → username;  patient incomplete → patient setup;  else → dashboard

async function routeSignedIn(user) {
    const id = ++routeId;
    stopDataListeners();
    setChecking("Loading your profile...");
    showView("checking");

    try {
        const caretaker = await withTimeout(getCaretaker(user.uid));
        if (id !== routeId) return;
        if (!caretaker || validateUsername(caretaker.username)) {
            setUser(user, null);
            showView("username");
            return;
        }
        setUser(user, caretaker.username);

        const patient = await withTimeout(getPatient());
        if (id !== routeId) return;
        if (!isPatientComplete(patient)) {
            fillPatientSetup(patient);
            showView("patientSetup");
            return;
        }

        showView("dashboard");
        startDataListeners();
    } catch (error) {
        if (id !== routeId) return;
        console.error("GripSense: loading profile failed", error);
        setUser(user, null);
        setChecking("Unable to load your profile. Please check your internet connection.", true);
    }
}

onRetryClick(() => {
    if (currentUser) routeSignedIn(currentUser);
});

watchAuth(user => {
    currentUser = user;
    if (signupInProgress) return;
    if (user) {
        routeSignedIn(user);
    } else {
        routeId++;
        stopDataListeners();
        setUser(null);
        showView("login");
    }
});

// ---- Login / signup ----

// Friendly text for Firebase Auth error codes.
function authErrorMessage(error, action = "Login") {
    switch (error.code) {
        case "auth/invalid-credential":
        case "auth/invalid-login-credentials":
        case "auth/wrong-password":
        case "auth/user-not-found":
            return "Incorrect email or password.";
        case "auth/invalid-email":
            return "Please enter a valid email address.";
        case "auth/email-already-in-use":
            return "An account with this email already exists. Please log in instead.";
        case "auth/weak-password":
            return "Password is too weak. Please choose a stronger password.";
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
            return `${action} failed (${error.code || error.message}).`;
    }
}

onFormSubmit("login", async ({ email, password }) => {
    setFormError("login", "");
    if (!email || !password) {
        setFormError("login", "Please enter your email and password.");
        return;
    }
    setFormBusy("login", true, "Logging in...");
    try {
        await login(email, password);
        // watchAuth routes to the next screen.
    } catch (error) {
        console.error("GripSense: login failed", error);
        setFormError("login", authErrorMessage(error));
    } finally {
        setFormBusy("login", false);
    }
});

onFormSubmit("signup", async values => {
    setFormError("signup", "");
    const errors = validateSignup(values);
    setFieldErrors("signup", errors);
    if (Object.keys(errors).length > 0) return;

    setFormBusy("signup", true, "Creating account...");
    signupInProgress = true;
    try {
        const { user } = await signup(values.email, values.password);
        try {
            await withTimeout(saveCaretaker(user.uid, { username: values.username, email: user.email }));
        } catch (error) {
            // The account exists; routing will ask for the username again.
            console.error("GripSense: saving caretaker profile failed", error);
        }
        signupInProgress = false;
        routeSignedIn(user);
    } catch (error) {
        signupInProgress = false;
        console.error("GripSense: signup failed", error);
        setFormError("signup", authErrorMessage(error, "Account creation"));
    } finally {
        setFormBusy("signup", false);
    }
});

onFormSubmit("username", async ({ username }) => {
    setFormError("username", "");
    const problem = validateUsername(username);
    setFieldErrors("username", problem ? { username: problem } : {});
    if (problem || !currentUser) return;

    setFormBusy("username", true, "Saving...");
    try {
        await withTimeout(saveCaretaker(currentUser.uid, { username, email: currentUser.email }));
        routeSignedIn(currentUser);
    } catch (error) {
        console.error("GripSense: saving caretaker profile failed", error);
        setFormError("username", `Could not save your username (${error.code || error.message}).`);
    } finally {
        setFormBusy("username", false);
    }
});

onFormSubmit("patientSetup", async values => {
    setFormError("patientSetup", "");
    const errors = validatePatientProfile(values);
    setFieldErrors("patientSetup", errors);
    if (Object.keys(errors).length > 0 || !currentUser) return;

    setFormBusy("patientSetup", true, "Saving...");
    try {
        await withTimeout(savePatientProfile({ ...values, age: Number(values.age) }));
        routeSignedIn(currentUser);
    } catch (error) {
        console.error("GripSense: saving patient profile failed", error);
        setFormError("patientSetup", `Could not save the patient profile (${error.code || error.message}).`);
    } finally {
        setFormBusy("patientSetup", false);
    }
});

onLogoutClick(() => {
    logout().catch(error => console.error("GripSense: logout failed", error));
});

// ---- Dashboard actions ----

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

showView("checking");
