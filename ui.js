// UI layer: render application state into the page.
// All text is inserted with textContent so stored values can never inject HTML.
import { NEED_MAPPINGS } from "./events.js";

// Fixed display format ("03 Oct 2026, 9:48 PM"); the time zone is still the browser's own.
const dateFormat = new Intl.DateTimeFormat("en-GB", { day: "2-digit", month: "short", year: "numeric" });
const timeFormat = new Intl.DateTimeFormat("en-US", { hour: "numeric", minute: "2-digit", hour12: true });

const el = {
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

function renderLatest(event, patientName) {
    const need = NEED_MAPPINGS[event.action];
    const card = el.latestCard;
    card.replaceChildren();
    card.className = `latest-card need-${need.color}`;
    card.dataset.action = event.action;

    card.append(
        make("div", "latest-need", `${need.icon} ${need.label.toUpperCase()}`),
        make("p", "latest-message", messageFor(event, patientName) ?? "Please set the patient name."),
        make("p", "latest-detail", `Maximum Pressure: ${event.maxPressure}`),
        make("p", "latest-detail", `Time: ${formatDateTime(event.timestamp)}`)
    );
}

function renderHistory(events, patientName) {
    const items = events.map(event => {
        const need = NEED_MAPPINGS[event.action];
        const item = make("li", `history-item need-${need.color}`);
        item.dataset.action = event.action;
        item.append(
            make("span", "history-need", `${need.icon} ${need.label}`),
            make("span", "history-message", messageFor(event, patientName) ?? ""),
            make("span", "history-pressure", `Pressure ${event.maxPressure}`),
            make("time", "history-time", formatDateTime(event.timestamp))
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
    el.notices.replaceChildren(...notices);
}

function showStatus(text, isError = false) {
    el.status.replaceChildren(...text.split("\n").map(line => make("p", "", line)));
    el.status.className = isError ? "status status-error" : "status";
    el.status.hidden = false;
}

export function render(state) {
    el.connection.textContent = state.connected ? "Connected to Firebase" : "Not connected";
    el.connection.className = state.connected ? "connection online" : "connection offline";

    renderNotices(state);

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
        renderLatest(state.events[0], state.patientName);
        renderHistory(state.events, state.patientName);
    }
}
