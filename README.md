# GripSense Caretaker Dashboard

Real-time web dashboard for GripSense patient alerts stored in Firebase Realtime Database.

## Setup

1. Copy `firebase-config.example.js` to `firebase-config.js` and fill in the Firebase Web App config
   (Firebase Console → Project settings → General → Your apps). This file is git-ignored.
2. From this folder, start a local server:

   ```sh
   python3 -m http.server 8000
   ```

3. Open <http://localhost:8000> (opening `index.html` directly will not work — ES modules need a server).

## Files

| File | Responsibility |
|---|---|
| `firebase.js` | Firebase initialization and real-time listeners |
| `events.js` | Action constants, need mapping, event validation and sorting |
| `profile.js` | Signup, username and patient-profile validation |
| `notify.js` | Alert sound patterns (Web Audio) and browser notifications |
| `ui.js` | Rendering latest alert, history, loading/empty/error states |
| `app.js` | Application state; connects data to the UI |
| `startup.js` | Intro video → logo startup screen, and the login animation (phone vs. laptop assets) |
| `manifest.json`, `sw.js` | PWA foundation: install metadata and a network-first service worker |

## Startup and media

On launch the intro video plays once (muted, with a Skip button), then the logo shows for about 1.5 s,
then the app: the dashboard if the caretaker is already logged in, otherwise login/signup with the login
animation. Firebase loads in the background meanwhile, and an unacknowledged patient alert ends the intro
immediately. Viewports up to 767 px wide use `videos/intro phone.mp4` and `videos/login mobile.mp4`;
wider ones (tablets, laptops) use the laptop videos. Only the videos needed are downloaded.

## Data contract

The ESP32 writes events to `/events` with `action` (`LIGHT_PRESS`, `HARD_PRESS`, `DOUBLE_SQUEEZE`),
`maxPressure` and `timestamp` (epoch ms). The dashboard maps actions to patient needs in `events.js`;
the shared patient profile is `/patient` (`name`, `age`, `gender`, `condition`, `updatedAt`), and each
caretaker's profile is `/caretakers/{uid}` (`username`, `email`, `createdAt`). Passwords are handled only
by Firebase Authentication.
