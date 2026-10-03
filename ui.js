// UI layer: render application state into the page.
// All text is inserted with textContent so stored values can never inject HTML.
import { NEED_MAPPINGS, selectLatestAlert } from "./events.js";

// Fixed display format ("03 Oct 2026, 9:48 PM"); the time zone is still the browser's own.
const dateFormat = new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short", year: "numeric" });
const timeFormat = new Intl.DateTimeFormat("en-US", { hour: "numeric", minute: "2-digit", hour12: true });

const el = {
    views: {
        checking: document.getElementById("checking-view"),
        login: document.getElementById("login-view"),
        dashboard: document.getElementById("dashboard-view")
    },
    userArea: document.getElementById("user-area"),
    userEmail: document.getElementById("user-email"),
    logoutButton: document.getElementById("logout-button"),
    loginForm: document.getElementById("login-form"),
    loginEmail: document.getElementById("login-email"),
    loginPassword: document.getElementById("login-password"),
    loginButton: document.getElementById("login-button"),
    loginError: document.getElementById("login-error"),
    patientForm: document.getElementById("patient-form"),
    patientInput: document.getElementById("patient-name"),
    patientSave: document.getElementById("patient-save"),
    patientFeedback: document.getElementById("patient-feedback"),
    connection: document.getElementById("connection"),
    notices: document.getElementById("notices"),
    status: document.getElementById("status"),
    latest: document.getElementById("latest"),
    latestCard: document.getElementById("latest-card"),
    history: document.getElementById("history"),
    historyList: document.getElementById("history-list")
};

// "03 Oct 2026, 9:48 PM" in the browser's local time zone.
export function formatDateTime(timestamp) {
    const date = new Date(timestamp);
    return `${dateFormat.format(date)}, ${timeFormat.format(date)}`;
}

// Time only when acknowledged on the same day as the event, otherwise full date.
function formatAckTime(eventTimestamp, ackTimestamp) {
    const sameDay = dateFormat.format(new Date(eventTimestamp)) === dateFormat.format(new Date(ackTimestamp));
    return sameDay ? timeFormat.format(new Date(ackTimestamp)) : formatDateTime(ackTimestamp);
}

function make(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

// Personalized message, or null when no patient name is configured yet.
function messageFor(event, patientName) {
    return patientName ? NEED_MAPPINGS[event.action].message(patientName) : null;
}

// "New" + Acknowledge button, or "Acknowledged · <time>".
function makeAckStatus(event, state) {
    const status = make("span", "ack-status");
    if (event.acknowledged) {
        status.classList.add("ack-done");
        status.textContent = event.acknowledgedAt
            ? `Acknowledged · ${formatAckTime(event.timestamp, event.acknowledgedAt)}`
            : "Acknowledged";
        return status;
    }
    const pending = state.acknowledging.has(event.id);
    const button = make("button", "button-ack", pending ? "Saving..." : "Acknowledge");
    button.type = "button";
    button.dataset.ackId = event.id;
    button.disabled = pending;
    status.append(make("span", "badge-new", "New"), button);
    return status;
}

function renderLatest(state) {
    const { event, allAcknowledged, moreUnacknowledged } = selectLatestAlert(state.events);
    const need = NEED_MAPPINGS[event.action];
    const card = el.latestCard;
    card.replaceChildren();
    card.className = `latest-card need-${need.color}${allAcknowledged ? " all-acknowledged" : ""}`;
    card.dataset.action = event.action;

    if (allAcknowledged) card.append(make("p", "latest-all-ack", "✓ All alerts acknowledged"));
    card.append(
        make("div", "latest-need", `${need.icon} ${need.label.toUpperCase()}`),
        make("p", "latest-message", messageFor(event, state.patientName) ?? "Please set the patient name."),
        make("p", "latest-detail", `Maximum Pressure: ${event.maxPressure}`),
        make("p", "latest-detail", `Time: ${formatDateTime(event.timestamp)}`),
        makeAckStatus(event, state)
    );
    if (moreUnacknowledged > 0) {
        card.append(make("p", "latest-more", `${moreUnacknowledged} more unacknowledged`));
    }
}

function renderHistory(state) {
    const items = state.events.map(event => {
        const need = NEED_MAPPINGS[event.action];
        const item = make("li", `history-item need-${need.color}${event.acknowledged ? " is-acknowledged" : ""}`);
        item.dataset.action = event.action;
        item.dataset.eventId = event.id;
        item.append(
            make("span", "history-need", `${need.icon} ${need.label}`),
            make("span", "history-message", messageFor(event, state.patientName) ?? ""),
            make("span", "history-pressure", `Pressure ${event.maxPressure}`),
            make("time", "history-time", formatDateTime(event.timestamp)),
            makeAckStatus(event, state)
        );
        item.querySelector("time").dateTime = new Date(event.timestamp).toISOString();
        return item;
    });
    el.historyList.replaceChildren(...items);
}

function renderNotices(state) {
    const notices = [];
    if (state.events.length > 0 && !state.connected) {
        notices.push(make("p", "notice notice-warning",
            "Connection to GripSense data lost. Alerts shown may be out of date."));
    }
    if (state.ackError) notices.push(make("p", "notice notice-error", state.ackError));
    el.notices.replaceChildren(...notices);
}

function showStatus(text, isError = false) {
    el.status.replaceChildren(...text.split("\n").map(line => make("p", "", line)));
    el.status.className = isError ? "status status-error" : "status";
    el.status.hidden = false;
}

// Show exactly one of: "checking" (login state unknown), "login", "dashboard".
export function showView(name) {
    for (const [key, view] of Object.entries(el.views)) view.hidden = key !== name;
    el.connection.hidden = name !== "dashboard";
}

export function setUser(user) {
    el.userArea.hidden = !user;
    el.userEmail.textContent = user ? user.email : "";
    el.loginPassword.value = "";
    el.patientInput.value = "";
    syncedPatientName = undefined;
    setPatientFeedback("");
}

export function onPatientSave(handler) {
    el.patientForm.addEventListener("submit", event => {
        event.preventDefault();
        handler(el.patientInput.value);
    });
    el.patientInput.addEventListener("input", () => setPatientFeedback(""));
}

export function setPatientBusy(busy) {
    el.patientSave.disabled = busy;
    el.patientSave.textContent = busy ? "Saving..." : "Save";
}

// type: "success" | "error"; an empty message hides the feedback.
export function setPatientFeedback(message, type = "success") {
    el.patientFeedback.textContent = message;
    el.patientFeedback.className = `form-feedback ${type}`;
    el.patientFeedback.hidden = !message;
}

// Fill the input with the stored name only when the stored name changes,
// so a caretaker's unsaved typing is not overwritten by unrelated updates.
let syncedPatientName;
function syncPatientInput(state) {
    if (!state.patientNameLoaded || state.patientName === syncedPatientName) return;
    syncedPatientName = state.patientName;
    el.patientInput.value = state.patientName ?? "";
}

export function onLoginSubmit(handler) {
    el.loginForm.addEventListener("submit", event => {
        event.preventDefault();
        handler(el.loginEmail.value.trim(), el.loginPassword.value);
    });
}

// One listener per container; buttons are re-created on every render.
export function onAcknowledgeClick(handler) {
    for (const container of [el.latestCard, el.historyList]) {
        container.addEventListener("click", event => {
            const button = event.target.closest("[data-ack-id]");
            if (button && !button.disabled) handler(button.dataset.ackId);
        });
    }
}

export function onLogoutClick(handler) {
    el.logoutButton.addEventListener("click", handler);
}

export function setLoginBusy(busy) {
    el.loginButton.disabled = busy;
    el.loginButton.textContent = busy ? "Logging in..." : "Login";
}

export function setLoginError(message) {
    el.loginError.textContent = message;
    el.loginError.hidden = !message;
}

export function render(state) {
    el.connection.textContent = state.connected ? "Connected to Firebase" : "Not connected";
    el.connection.className = state.connected ? "connection online" : "connection offline";

    renderNotices(state);
    syncPatientInput(state);

    const hasEvents = state.events.length > 0;
    el.latest.hidden = !hasEvents;
    el.history.hidden = !hasEvents;

    if (state.error && !hasEvents) {
        showStatus("Unable to connect to GripSense data.\nPlease check your internet connection.", true);
    } else if (!state.eventsLoaded) {
        showStatus("Loading GripSense events...");
    } else if (!hasEvents) {
        showStatus("No patient alerts yet.");
    } else {
        el.status.hidden = true;
        renderLatest(state);
        renderHistory(state);
    }
}
