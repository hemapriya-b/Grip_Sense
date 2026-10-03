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
| `ui.js` | Rendering latest alert, history, loading/empty/error states |
| `app.js` | Application state; connects data to the UI |

## Data contract

The ESP32 writes events to `/events` with `action` (`LIGHT_PRESS`, `HARD_PRESS`, `DOUBLE_SQUEEZE`),
`maxPressure` and `timestamp` (epoch ms). The dashboard maps actions to patient needs in `events.js`;
the shared patient profile is `/patient` (`name`, `age`, `gender`, `condition`, `updatedAt`), and each
caretaker's profile is `/caretakers/{uid}` (`username`, `email`, `createdAt`). Passwords are handled only
by Firebase Authentication.
