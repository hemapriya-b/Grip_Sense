// UI layer: render application state into the page.
// All text is inserted with textContent so stored values can never inject HTML.
import { NEED_MAPPINGS, selectLatestAlert } from "./events.js";

// Fixed display format ("03 Oct 2026, 9:48 PM"); the time zone is still the browser's own.
const dateFormat = new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short", year: "numeric" });
const longDateFormat = new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "long", year: "numeric" });
const timeFormat = new Intl.DateTimeFormat("en-US", { hour: "numeric", minute: "2-digit", hour12: true });

const THEME_KEY = "gripsense-theme";
const $ = id => document.getElementById(id);

const el = {
    views: {
        checking: $("checking-view"),
        login: $("login-view"),
        signup: $("signup-view"),
        username: $("username-view"),
        patientSetup: $("patient-setup-view"),
        dashboard: $("dashboard-view")
    },
    forms: {
        login: { form: $("login-form"), button: $("login-button"), error: $("login-error") },
        signup: { form: $("signup-form"), button: $("signup-button"), error: $("signup-error") },
        username: { form: $("username-form"), button: $("username-button"), error: $("username-error") },
        patientSetup: { form: $("patient-setup-form"), button: $("patient-setup-button"), error: $("patient-setup-error") }
    },
    checkingMessage: $("checking-message"),
    retryButton: $("retry-button"),
    themeToggle: $("theme-toggle"),
    userArea: $("user-area"),
    userName: $("user-name"),
    logoutButton: $("logout-button"),
    patientForm: $("patient-form"),
    patientInput: $("patient-name"),
    patientSave: $("patient-save"),
    patientFeedback: $("patient-feedback"),
    connection: $("connection"),
    notices: $("notices"),
    status: $("status"),
    latest: $("latest"),
    latestCard: $("latest-card"),
    history: $("history"),
    historyList: $("history-list"),
    eventDialog: $("event-dialog"),
    eventDetail: $("event-detail"),
    alert: {
        screen: $("alert-screen"),
        tag: $("alert-tag"),
        icon: $("alert-icon"),
        label: $("alert-label"),
        message: $("alert-message"),
        meta: $("alert-meta"),
        ack: $("alert-ack"),
        more: $("alert-more"),
        sound: $("alert-sound")
    },
    soundStatus: $("sound-status"),
    enableSound: $("enable-sound"),
    notificationStatus: $("notification-status"),
    enableNotifications: $("enable-notifications")
};

const APP_TITLE = document.title;
let shownAlertId = null;

// Last rendered dashboard state, so the detail dialog can re-render on updates.
let lastState = null;
let openEventId = null;

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

function makeAckButton(event, state, className = "button-ack") {
    const pending = state.acknowledging.has(event.id);
    const button = make("button", className, pending ? "Saving..." : "Acknowledge");
    button.type = "button";
    button.dataset.ackId = event.id;
    button.disabled = pending;
    return button;
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
    status.append(make("span", "badge-new", "New"), makeAckButton(event, state));
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

        // The whole row opens details; this button makes it keyboard-accessible.
        const open = make("button", "history-open");
        open.type = "button";
        open.setAttribute("aria-haspopup", "dialog");
        const time = make("time", "", formatDateTime(event.timestamp));
        time.dateTime = new Date(event.timestamp).toISOString();
        const meta = make("span", "history-meta");
        meta.append(time, ` · Pressure ${event.maxPressure}`);
        open.append(
            make("span", "history-need", `${need.icon} ${need.label}`),
            make("span", "history-message", messageFor(event, state.patientName) ?? ""),
            meta
        );

        item.append(open, makeAckStatus(event, state));
        return item;
    });
    el.historyList.replaceChildren(...items);
}

function detailRow(label, value) {
    const row = make("div", "detail-row");
    const dd = make("dd");
    if (value instanceof Node) dd.append(value); else dd.textContent = value;
    row.append(make("dt", "", label), dd);
    return row;
}

// Read-only view of one event; only the Acknowledge button writes anything.
function renderEventDetail() {
    const event = lastState?.events.find(e => e.id === openEventId);
    if (!event) {
        closeEventDetail();
        return;
    }
    const need = NEED_MAPPINGS[event.action];
    const date = new Date(event.timestamp);

    const header = make("div", `detail-header need-${need.color}`);
    const title = make("h3", "detail-title", `${need.icon} ${need.label}`);
    title.id = "event-detail-title";
    header.append(title, make("p", "detail-message", messageFor(event, lastState.patientName) ?? "Please set the patient name."));

    const status = make("span", event.acknowledged ? "detail-status done" : "detail-status new",
        event.acknowledged ? "● Acknowledged" : "● NEW");

    const list = make("dl", "detail-list");
    list.append(
        detailRow("Event", event.action),
        detailRow("Maximum Pressure", String(event.maxPressure)),
        detailRow("Date", longDateFormat.format(date)),
        detailRow("Time", timeFormat.format(date)),
        detailRow("Status", status)
    );
    if (event.acknowledged && event.acknowledgedAt) {
        list.append(detailRow("Acknowledged at", formatAckTime(event.timestamp, event.acknowledgedAt)));
    }

    el.eventDetail.replaceChildren(header, list);
    if (lastState.ackError) el.eventDetail.append(make("p", "notice notice-error", lastState.ackError));
    if (!event.acknowledged) {
        el.eventDetail.append(makeAckButton(event, lastState, "button-ack button-ack-large"));
    }
}

function openEventDetail(eventId) {
    openEventId = eventId;
    renderEventDetail();
    if (openEventId && !el.eventDialog.open) el.eventDialog.showModal();
}

function closeEventDetail() {
    openEventId = null;
    if (el.eventDialog.open) el.eventDialog.close();
}

el.eventDialog.addEventListener("close", () => { openEventId = null; });
// Clicking the dimmed backdrop (the dialog element itself) closes it.
el.eventDialog.addEventListener("click", event => {
    if (event.target === el.eventDialog) closeEventDetail();
});
el.historyList.addEventListener("click", event => {
    if (event.target.closest("[data-ack-id]")) return;
    const item = event.target.closest("[data-event-id]");
    if (item) openEventDetail(item.dataset.eventId);
});

// ---- Full-screen patient alert ----
// Shown whenever an unacknowledged event exists; it overrides the light/dark theme.

function hideAlertScreen() {
    el.alert.screen.hidden = true;
    delete document.body.dataset.alert;
    document.title = APP_TITLE;
    shownAlertId = null;
}

function renderAlertScreen(state) {
    const { event, allAcknowledged, moreUnacknowledged } = selectLatestAlert(state.events);
    if (!state.eventsLoaded || allAcknowledged || !event) {
        hideAlertScreen();
        return;
    }
    const need = NEED_MAPPINGS[event.action];
    const a = el.alert;

    a.screen.hidden = false;
    document.body.dataset.alert = need.color;
    a.tag.textContent = state.newThisSession.has(event.id) ? "NEW ALERT" : "ACTIVE ALERT";
    a.icon.textContent = need.icon;
    a.label.textContent = need.label.toUpperCase();
    a.message.textContent = messageFor(event, state.patientName) ?? "Please set the patient name.";
    a.meta.textContent = `${formatDateTime(event.timestamp)} · Pressure ${event.maxPressure}`;

    const pending = state.acknowledging.has(event.id);
    a.ack.dataset.ackId = event.id;
    a.ack.disabled = pending;
    a.ack.textContent = pending ? "SAVING..." : "ACKNOWLEDGE";

    a.more.textContent = state.ackError
        ?? (moreUnacknowledged > 0 ? `${moreUnacknowledged} more unacknowledged alert${moreUnacknowledged > 1 ? "s" : ""}` : "");
    a.more.classList.toggle("is-error", Boolean(state.ackError));
    a.sound.hidden = state.audio !== "locked";

    document.title = `${need.icon} ${need.label.toUpperCase()} — GripSense`;

    // A different alert came to the front: make sure nothing covers it.
    if (shownAlertId !== event.id) {
        shownAlertId = event.id;
        closeEventDetail();
        a.ack.focus();
    }
}

function renderAlertSettings(state) {
    const sound = { running: "On", locked: "Off", unsupported: "Not supported in this browser" };
    el.soundStatus.textContent = sound[state.audio];
    el.enableSound.hidden = state.audio !== "locked";

    const notifications = {
        granted: "On (when GripSense is in the background)",
        denied: "Blocked — allow notifications for this site in browser settings",
        default: "Off",
        unsupported: "Not supported in this browser"
    };
    el.notificationStatus.textContent = notifications[state.notifications];
    el.enableNotifications.hidden = state.notifications !== "default";
}

export function onEnableSound(handler) {
    el.enableSound.addEventListener("click", handler);
    el.alert.sound.addEventListener("click", handler);
}

export function onEnableNotifications(handler) {
    el.enableNotifications.addEventListener("click", handler);
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

// ---- Theme ----

function applyThemeLabel() {
    const dark = document.documentElement.dataset.theme === "dark";
    el.themeToggle.textContent = dark ? "☀️ Light" : "🌙 Dark";
    el.themeToggle.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
}

el.themeToggle.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem(THEME_KEY, next);
    applyThemeLabel();
});
applyThemeLabel();

// ---- Views and forms ----

// Show exactly one view: checking, login, signup, username, patientSetup, dashboard.
export function showView(name) {
    for (const [key, view] of Object.entries(el.views)) view.hidden = key !== name;
    el.connection.hidden = name !== "dashboard";
    if (name !== "dashboard") {
        closeEventDetail();
        hideAlertScreen();
    }
    for (const input of document.querySelectorAll("input[type=password]")) input.value = "";
    const firstInput = el.views[name].querySelector("input, select, textarea");
    if (firstInput && name !== "dashboard") firstInput.focus();
}

export function setChecking(message, showRetry = false) {
    el.checkingMessage.textContent = message;
    el.views.checking.classList.toggle("status-error", showRetry);
    el.retryButton.hidden = !showRetry;
}

export function onRetryClick(handler) {
    el.retryButton.addEventListener("click", handler);
}

$("show-signup").addEventListener("click", () => { clearForm("signup"); showView("signup"); });
$("show-login").addEventListener("click", () => { clearForm("login"); showView("login"); });

function fieldsOf(name) {
    return el.forms[name].form.querySelectorAll("[data-field]");
}

// Calls handler({ fieldName: value, ... }) when the named form is submitted.
export function onFormSubmit(name, handler) {
    el.forms[name].form.addEventListener("submit", event => {
        event.preventDefault();
        const values = {};
        for (const field of fieldsOf(name)) {
            values[field.dataset.field] = field.type === "password" ? field.value : field.value.trim();
        }
        handler(values);
    });
}

// errors: { fieldName: message }. Fields without an entry are cleared.
export function setFieldErrors(name, errors) {
    let first = null;
    for (const field of fieldsOf(name)) {
        const message = errors[field.dataset.field] ?? "";
        const errorEl = $(`${field.id}-error`);
        if (errorEl) {
            errorEl.textContent = message;
            errorEl.hidden = !message;
        }
        field.toggleAttribute("aria-invalid", Boolean(message));
        if (message && !first) first = field;
    }
    first?.focus();
}

export function setFormError(name, message) {
    const { error } = el.forms[name];
    error.textContent = message;
    error.hidden = !message;
}

export function setFormBusy(name, busy, busyText) {
    const { button } = el.forms[name];
    button.dataset.idleText ??= button.textContent;
    button.disabled = busy;
    button.textContent = busy ? busyText : button.dataset.idleText;
}

function clearForm(name) {
    setFieldErrors(name, {});
    setFormError(name, "");
}

// Pre-fills the patient setup form with whatever is already stored (e.g. an old /patient/name).
export function fillPatientSetup(patient) {
    clearForm("patientSetup");
    for (const field of fieldsOf("patientSetup")) {
        const value = patient?.[field.dataset.field];
        field.value = value === undefined || value === null ? "" : String(value);
    }
}

export function setUser(user, username) {
    el.userArea.hidden = !user;
    el.userName.textContent = user ? (username || user.email) : "";
    el.patientInput.value = "";
    syncedPatientName = undefined;
    setPatientFeedback("");
}

// ---- Dashboard patient name form (Step 8) ----

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

// One listener per container; buttons are re-created on every render.
export function onAcknowledgeClick(handler) {
    for (const container of [el.latestCard, el.historyList, el.eventDetail, el.alert.screen]) {
        container.addEventListener("click", event => {
            const button = event.target.closest("[data-ack-id]");
            if (button && !button.disabled) handler(button.dataset.ackId);
        });
    }
}

export function onLogoutClick(handler) {
    el.logoutButton.addEventListener("click", handler);
}

export function render(state) {
    lastState = state;
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
    if (openEventId) renderEventDetail();
    renderAlertSettings(state);
    renderAlertScreen(state);
}
