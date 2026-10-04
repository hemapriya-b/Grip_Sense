// Startup experience: intro video → logo → app, plus the login animation.
// The intro plays with its original sound; the login animation is always muted.
// Loaded on its own from index.html so the intro plays even while (or if) Firebase
// is still loading. Phone-sized viewports get the portrait videos; tablets and
// laptops get the landscape ones. Only the videos actually needed are downloaded.

const PHONE_QUERY = matchMedia("(max-width: 767px)");
const MEDIA = {
    phone: { intro: "videos/intro%20phone.mp4", login: "videos/login%20mobile.mp4" },
    laptop: { intro: "videos/intro%20laptop.mp4", login: "videos/login%20laptop.mp4" }
};
const LOGO_MS = 1500;
const FADE_MS = 400;
// Safety net: never hold the app on the intro longer than this, even if the video stalls.
const INTRO_MAX_MS = 20000;

const $ = id => document.getElementById(id);
const screen = $("startup");
const introVideo = $("intro-video");
const introPlay = $("intro-play");
const introSkip = $("intro-skip");
const loginVideo = $("login-video");
// The app underneath can't be focused or clicked while the startup screen covers it.
const appParts = [document.querySelector(".header"), document.querySelector(".main")];

let stage = "intro";            // intro → logo → done
let introTimer = null;
let logoTimer = null;
let loginDevice = null;         // which login video is loaded: "phone" | "laptop"
let loginWanted = false;        // login/signup is the current view
const doneHandlers = [];

const currentDevice = () => (PHONE_QUERY.matches ? "phone" : "laptop");

export function isStartupDone() {
    return stage === "done";
}

// Runs handler once the startup screen is gone (immediately if it already is).
export function onStartupDone(handler) {
    if (isStartupDone()) handler(); else doneHandlers.push(handler);
}

function playIntro() {
    clearTimeout(introTimer);
    introTimer = setTimeout(showLogo, INTRO_MAX_MS);
    introVideo.play().catch(error => {
        if (stage !== "intro") return;
        clearTimeout(introTimer);
        if (error.name !== "NotAllowedError") {
            console.warn("GripSense: intro video could not play", error);
            showLogo();
            return;
        }
        // Browsers usually block autoplay *with sound* until the first tap.
        // Never fall back to muted: wait for "Tap to start GripSense" instead.
        introPlay.hidden = false;
        introPlay.focus();
    });
}

// Stop the intro and free its memory; it is never shown again in this session.
function releaseIntro() {
    clearTimeout(introTimer);
    introVideo.pause();
    introVideo.removeAttribute("src");
    introVideo.load();
}

function showLogo() {
    if (stage !== "intro") return;
    stage = "logo";
    releaseIntro();
    screen.dataset.stage = "logo";
    logoTimer = setTimeout(finishStartup, LOGO_MS);
}

function finishStartup({ immediate = false } = {}) {
    if (stage === "done") return;
    stage = "done";
    releaseIntro();
    clearTimeout(logoTimer);
    appParts.forEach(part => { part.inert = false; });
    if (immediate) {
        screen.hidden = true;
    } else {
        screen.classList.add("is-leaving");
        setTimeout(() => { screen.hidden = true; }, FADE_MS);
    }
    if (loginWanted) playLoginAnimation();
    doneHandlers.splice(0).forEach(handler => handler());
}

// A waiting patient alert has priority over the intro: end it right away.
export function endStartup() {
    finishStartup({ immediate: true });
}

// ---- Login animation (on the login/signup screen) ----
// Plays once, muted, and then stays on its final frame. Switching between
// Login and Signup keeps the same element, so it does not restart.

function playLoginAnimation() {
    if (loginVideo.ended) return;
    loginVideo.play().catch(() => {
        // Autoplay blocked: the first frame stays as a still image; the form still works.
    });
}

export function showLoginAnimation() {
    loginWanted = true;
    const device = currentDevice();
    if (loginDevice !== device) {
        loginDevice = device;
        loginVideo.src = MEDIA[device].login;
    }
    // Under the startup screen it only preloads; it plays once the screen is gone.
    if (isStartupDone()) playLoginAnimation();
}

export function hideLoginAnimation() {
    loginWanted = false;
    loginVideo.pause();
}

// Rotating a phone or resizing a window across the breakpoint switches to the matching video.
PHONE_QUERY.addEventListener("change", () => {
    if (loginWanted) showLoginAnimation();
});

// ---- Start ----

appParts.forEach(part => { part.inert = true; });
screen.dataset.device = currentDevice();
introVideo.muted = false;
loginVideo.muted = true;
introVideo.addEventListener("ended", showLogo);
introVideo.addEventListener("error", () => {
    if (introVideo.getAttribute("src")) console.warn("GripSense: intro video failed to load", introVideo.error);
    showLogo();
});
loginVideo.addEventListener("error", () => {
    if (loginVideo.getAttribute("src")) console.warn("GripSense: login video failed to load", loginVideo.error);
});
introSkip.addEventListener("click", showLogo);
introPlay.addEventListener("click", () => {
    introPlay.hidden = true;
    playIntro();
});
introVideo.src = MEDIA[currentDevice()].intro;
playIntro();
