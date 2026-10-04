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
import { parseEvents, NEED_MAPPINGS } from "./events.js";
import {
    audioStatus,
    unlockAudio,
    playAlertSound,
    stopAlertSound,
    playingEventId,
    notificationStatus,
    requestNotificationPermission,
    isAppInView,
    showSystemNotification
} from "./notify.js";
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
    onAcknowledgeClick,
    onEnableSound,
    onEnableNotifications
} from "./ui.js";
import { endStartup } from "./startup.js";

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
    ackError: null,
    newThisSession: new Set(),  // unacknowledged events that arrived after the dashboard loaded
    audio: audioStatus(),
    notifications: notificationStatus()
});

let state = initialState();

// Per dashboard session: every event ID seen so far, and events whose sound has
// already been played. Firebase re-sends the whole list on every change, so these
// guarantee each event alerts at most once.
let seenEventIds = new Set();
let soundPlayedIds = new Set();
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

    seenEventIds = new Set();
    soundPlayedIds = new Set();

    stopListeners = [
        listenToEvents(raw => {
            clearTimeout(loadTimer);
            const events = parseEvents(raw);
            const isInitialLoad = !state.eventsLoaded;

            // Events present at load are not "new": no sound, no notification.
            const arrivals = events.filter(e => !seenEventIds.has(e.id));
            arrivals.forEach(e => seenEventIds.add(e.id));
            const newAlerts = isInitialLoad ? [] : arrivals.filter(e => !e.acknowledged);
            newAlerts.forEach(e => state.newThisSession.add(e.id));

            update({ events, eventsLoaded: true, error: null });
            // A waiting patient alert has priority over the intro/logo: show it now.
            if (events.some(e => !e.acknowledged)) endStartup();
            stopSoundIfResolved();
            if (newAlerts.length > 0) announceNewAlerts(newAlerts);
        }, handleError),

        listenToPatientName(name => {
            const trimmed = typeof name === "string" ? name.trim() : "";
            update({ patientName: trimmed || null, patientNameLoaded: true });
        }, handleError),

        listenToConnection(connected => update({ connected }))
    ];
}

// ---- New-alert announcements ----

function alertText(event) {
    const need = NEED_MAPPINGS[event.action];
    return `${need.icon} ${state.patientName ? need.message(state.patientName) : `${need.label} alert`}`;
}

// newAlerts is newest first. The full-screen alert itself is driven by render().
function announceNewAlerts(newAlerts) {
    playSoundOnce(newAlerts[0]);
    if (!isAppInView()) {
        for (const event of newAlerts) showSystemNotification(event.id, alertText(event));
    }
}

async function playSoundOnce(event) {
    if (soundPlayedIds.has(event.id)) return;
    // Try to start audio; browsers that still require a click keep it locked.
    if (audioStatus() !== "running") await unlockAudio();
    if (soundPlayedIds.has(event.id)) return;
    if (playAlertSound(event.id, event.action)) soundPlayedIds.add(event.id);
    // Locked → shows "Enable Alert Sound"; running → settings show "On".
    if (audioStatus() !== state.audio) update({ audio: audioStatus() });
}

// Stop the pattern as soon as its event is acknowledged (here or by another caretaker).
function stopSoundIfResolved() {
    const id = playingEventId();
    if (id && !state.events.some(e => e.id === id && !e.acknowledged)) stopAlertSound();
}

// Any click/tap/key press lets the browser start audio (including the
// "Enable Alert Sound" button). If the current alert arrived while sound was
// blocked, it plays now — still at most once per event.
async function unlockAudioFromGesture() {
    if (audioStatus() === "running") return;
    const status = await unlockAudio();
    if (status !== state.audio) update({ audio: status });
    const active = state.events.find(e => !e.acknowledged);
    if (status === "running" && active && state.newThisSession.has(active.id)) playSoundOnce(active);
}
document.addEventListener("click", unlockAudioFromGesture, true);
document.addEventListener("keydown", unlockAudioFromGesture, true);
onEnableSound(unlockAudioFromGesture);

onEnableNotifications(async () => {
    const status = await requestNotificationPermission();
    update({ notifications: status });
});

// Permission can be changed in browser settings while the page is open.
document.addEventListener("visibilitychange", () => {
    if (state.notifications !== notificationStatus()) update({ notifications: notificationStatus() });
});

function stopDataListeners() {
    stopAlertSound();
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
