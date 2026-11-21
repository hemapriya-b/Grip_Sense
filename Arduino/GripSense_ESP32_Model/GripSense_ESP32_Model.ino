#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>

// ================= WIFI =================
#include "secrets.h"
// ================= FIREBASE =================
const char* firebaseURL =
  "https://gripsense-7d636-default-rtdb.firebaseio.com/events.json";

// ================= PINS =================
const int FSR_PIN = 34;

const int RED_PIN = 25;
const int GREEN_PIN = 26;
const int BLUE_PIN = 27;

const int BUZZER_PIN = 23;

// ================= THRESHOLDS =================
const int NO_PRESSURE = 10;
const int HARD_PRESS = 700;

// ================= TIMING =================
const unsigned long DETECTION_TIME = 2000;
const unsigned long LED_TIME = 30000;

// Green: 10 sec ON / 5 sec OFF
const unsigned long LIGHT_BEEP_TIME = 10000;
const unsigned long LIGHT_GAP = 5000;

// Red: 10 sec ON / 2 sec OFF
const unsigned long HARD_BEEP_TIME = 10000;
const unsigned long HARD_GAP = 2000;

// Blue: 0.3 sec ON / 2 sec OFF
const unsigned long DOUBLE_BEEP_TIME = 300;
const unsigned long DOUBLE_GAP = 2000;

// ================= VARIABLES =================
bool detecting = false;
bool ledActive = false;

unsigned long detectionStart = 0;
unsigned long ledStart = 0;

int maxPressure = 0;

int consecutiveZeros = 0;
bool doubleSqueezeDetected = false;

int actionType = 0;

// =================================================
// SEND ACTION TO FIREBASE
// =================================================
void sendToFirebase(String action, int pressure) {

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi not connected. Firebase upload skipped.");
    return;
  }

  WiFiClientSecure client;
  client.setInsecure();

  HTTPClient http;

  Serial.println();
  Serial.println("Sending action to Firebase...");

  http.begin(client, firebaseURL);
  http.addHeader("Content-Type", "application/json");

  // Firebase server timestamp
  String jsonData =
    "{\"action\":\"" + action +
    "\",\"maxPressure\":" + String(pressure) +
    ",\"timestamp\":{\".sv\":\"timestamp\"}}";

  int httpCode = http.POST(jsonData);

  Serial.print("Firebase HTTP Code: ");
  Serial.println(httpCode);

  if (httpCode > 0) {

    String response = http.getString();

    Serial.print("Firebase Response: ");
    Serial.println(response);

    if (httpCode == 200) {
      Serial.println("Firebase upload SUCCESS!");
    }

  } else {

    Serial.print("Firebase Error: ");
    Serial.println(http.errorToString(httpCode));
  }

  http.end();
}

// =================================================
// SET LED
// =================================================
void setLED(int action) {

  digitalWrite(RED_PIN, LOW);
  digitalWrite(GREEN_PIN, LOW);
  digitalWrite(BLUE_PIN, LOW);

  if (action == 1) {
    // GREEN = light press
    digitalWrite(GREEN_PIN, HIGH);
  }

  else if (action == 2) {
    // RED = hard press
    digitalWrite(RED_PIN, HIGH);
  }

  else if (action == 3) {
    // BLUE = double squeeze
    digitalWrite(BLUE_PIN, HIGH);
  }
}

// =================================================
// TURN EVERYTHING OFF
// =================================================
void turnEverythingOff() {

  digitalWrite(RED_PIN, LOW);
  digitalWrite(GREEN_PIN, LOW);
  digitalWrite(BLUE_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
}

// =================================================
// SETUP
// =================================================
void setup() {

  Serial.begin(115200);

  pinMode(RED_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  pinMode(BLUE_PIN, OUTPUT);

  pinMode(BUZZER_PIN, OUTPUT);

  turnEverythingOff();

  Serial.println();
  Serial.println("================================");
  Serial.println("GripSense + Firebase");
  Serial.println("================================");

  // ---------- Wi-Fi ----------
  WiFi.begin(ssid, password);

  Serial.print("Connecting to Wi-Fi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("Wi-Fi connected!");

  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  Serial.println();
  Serial.println("GripSense READY");
}

// =================================================
// MAIN LOOP
// =================================================
void loop() {

  int pressure = analogRead(FSR_PIN);

  // Show pressure continuously while detecting
  if (detecting) {
    Serial.print("Pressure: ");
    Serial.println(pressure);
  }

  // =================================================
  // START DETECTION
  // =================================================
  if (!detecting && !ledActive && pressure > NO_PRESSURE) {

    detecting = true;

    detectionStart = millis();

    maxPressure = pressure;

    consecutiveZeros = 0;

    doubleSqueezeDetected = false;

    Serial.println();
    Serial.println(">>> Detection started");
  }

  // =================================================
  // DURING DETECTION
  // =================================================
  if (detecting) {

    // Record maximum pressure
    if (pressure > maxPressure) {
      maxPressure = pressure;
    }

    // Check for two consecutive ZERO readings
    if (pressure == 0) {

      consecutiveZeros++;

      if (consecutiveZeros >= 2) {
        doubleSqueezeDetected = true;
      }

    } else {

      consecutiveZeros = 0;
    }

    // =================================================
    // END DETECTION WINDOW
    // =================================================
    if (millis() - detectionStart >= DETECTION_TIME) {

      detecting = false;

      Serial.println();
      Serial.println(">>> Detection finished");

      // ---------------------------------------------
      // DOUBLE SQUEEZE
      // ---------------------------------------------
      if (doubleSqueezeDetected) {

        actionType = 3;

        Serial.println("ACTION: DOUBLE SQUEEZE");
        Serial.print("Maximum pressure: ");
        Serial.println(maxPressure);

        setLED(actionType);

        sendToFirebase("DOUBLE_SQUEEZE", maxPressure);
      }

      // ---------------------------------------------
      // LIGHT PRESS
      // ---------------------------------------------
      else if (maxPressure <= HARD_PRESS) {

        actionType = 1;

        Serial.println("ACTION: LIGHT PRESS");
        Serial.print("Maximum pressure: ");
        Serial.println(maxPressure);

        setLED(actionType);

        sendToFirebase("LIGHT_PRESS", maxPressure);
      }

      // ---------------------------------------------
      // HARD PRESS
      // ---------------------------------------------
      else {

        actionType = 2;

        Serial.println("ACTION: HARD PRESS");
        Serial.print("Maximum pressure: ");
        Serial.println(maxPressure);

        setLED(actionType);

        sendToFirebase("HARD_PRESS", maxPressure);
      }

      // Start 30-second alert period
      ledActive = true;
      ledStart = millis();

      Serial.println("30-second alert started.");
    }
  }

  // =================================================
  // 30-SECOND LED + BUZZER PERIOD
  // =================================================
  if (ledActive) {

    unsigned long elapsed = millis() - ledStart;

    // ---------------------------------------------
    // AFTER 30 SECONDS
    // ---------------------------------------------
    if (elapsed >= LED_TIME) {

      turnEverythingOff();

      ledActive = false;
      actionType = 0;

      Serial.println();
      Serial.println(">>> Alert finished");
      Serial.println(">>> GripSense READY");
      Serial.println();
    }

    // ---------------------------------------------
    // BUZZER PATTERNS
    // ---------------------------------------------
    else {

      bool buzzerOn = false;

      if (actionType == 1) {

        // GREEN:
        // 10 sec ON / 5 sec OFF

        unsigned long cycle =
          LIGHT_BEEP_TIME + LIGHT_GAP;

        unsigned long position =
          elapsed % cycle;

        buzzerOn =
          position < LIGHT_BEEP_TIME;
      }

      else if (actionType == 2) {

        // RED:
        // 10 sec ON / 2 sec OFF

        unsigned long cycle =
          HARD_BEEP_TIME + HARD_GAP;

        unsigned long position =
          elapsed % cycle;

        buzzerOn =
          position < HARD_BEEP_TIME;
      }

      else if (actionType == 3) {

        // BLUE:
        // 0.3 sec ON / 2 sec OFF

        unsigned long cycle =
          DOUBLE_BEEP_TIME + DOUBLE_GAP;

        unsigned long position =
          elapsed % cycle;

        buzzerOn =
          position < DOUBLE_BEEP_TIME;
      }

      digitalWrite(BUZZER_PIN, buzzerOn ? HIGH : LOW);
    }
  }

  delay(10);
}
