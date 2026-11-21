#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <math.h>
#include <stdint.h>

extern "C" {
  int gs_predict_from_adc(const uint16_t adc[150]);
  int gs_extract_features(const uint16_t adc[150], double out[2]);
  const char *gs_label_name(int label);
}

// Add Wi-Fi credentials locally; do not publish real credentials.
const char* ssid = "SWEET HOME 5G";
const char* password = "Net Illa";
const char* firebaseURL = "https://gripsense-7d636-default-rtdb.firebaseio.com/events.json";

const int FSR_PIN = 34, RED_PIN = 25, GREEN_PIN = 26, BLUE_PIN = 27, BUZZER_PIN = 23;
const int NO_PRESSURE = 10, HARD_PRESS = 1000;
const unsigned long DETECTION_TIME = 2000, LED_TIME = 30000;
const unsigned long LIGHT_BEEP_TIME = 10000, LIGHT_GAP = 5000;
const unsigned long HARD_BEEP_TIME = 10000, HARD_GAP = 2000;
const unsigned long DOUBLE_BEEP_TIME = 300, DOUBLE_GAP = 2000;
const int ML_SAMPLES = 150;
const unsigned long ML_SAMPLE_INTERVAL_MS = 20;

uint16_t adcWindow[ML_SAMPLES];
bool detecting = false, ledActive = false;
unsigned long detectionStart = 0, ledStart = 0;
int maxPressure = 0, consecutiveZeros = 0, actionType = 0;
bool doubleSqueezeDetected = false;

void sendToFirebase(String action, int pressure) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi not connected. Firebase upload skipped.");
    return;
  }
  WiFiClientSecure client;
  client.setInsecure(); // Preserves original sketch behavior.
  HTTPClient http;
  Serial.println("\nSending action to Firebase...");
  if (!http.begin(client, firebaseURL)) {
    Serial.println("Firebase request could not be initialized.");
    return;
  }
  http.addHeader("Content-Type", "application/json");
  String jsonData = "{\"action\":\"" + action + "\",\"maxPressure\":" +
    String(pressure) + ",\"timestamp\":{\".sv\":\"timestamp\"}}";
  int httpCode = http.POST(jsonData);
  Serial.print("Firebase HTTP Code: "); Serial.println(httpCode);
  if (httpCode > 0) {
    Serial.print("Firebase Response: "); Serial.println(http.getString());
    if (httpCode == 200) Serial.println("Firebase upload SUCCESS!");
  } else {
    Serial.print("Firebase Error: "); Serial.println(http.errorToString(httpCode));
  }
  http.end();
}

void setLED(int action) {
  digitalWrite(RED_PIN, LOW); digitalWrite(GREEN_PIN, LOW); digitalWrite(BLUE_PIN, LOW);
  if (action == 1) digitalWrite(GREEN_PIN, HIGH);
  else if (action == 2) digitalWrite(RED_PIN, HIGH);
  else if (action == 3) digitalWrite(BLUE_PIN, HIGH);
}
void turnEverythingOff() {
  digitalWrite(RED_PIN, LOW); digitalWrite(GREEN_PIN, LOW);
  digitalWrite(BLUE_PIN, LOW); digitalWrite(BUZZER_PIN, LOW);
}

// Returns false only for invalid/non-finite model-path output.
// The model API has no confidence score, so valid-but-wrong predictions cannot
// automatically be identified as failures.
bool runTinyML(int &prediction, double features[2]) {
  gs_extract_features(adcWindow, features);
  if (!isfinite(features[0]) || !isfinite(features[1])) return false;
  prediction = gs_predict_from_adc(adcWindow);
  return prediction >= 0 && prediction <= 3;
}

void finishAction(int action, const char* source) {
  // action 0 = NO_PRESS/no alert; 1 light, 2 hard, 3 double squeeze.
  if (action == 0) {
    Serial.println("TinyML predicted NO_PRESS; no alert action started.");
    Serial.println(">>> GripSense READY");
    return;
  }
  actionType = action;
  if (action == 1) Serial.println("ACTION: LIGHT PRESS");
  else if (action == 2) Serial.println("ACTION: HARD PRESS");
  else Serial.println("ACTION: DOUBLE SQUEEZE");
  Serial.print("Classification source: "); Serial.println(source);
  Serial.print("Maximum pressure: "); Serial.println(maxPressure);
  setLED(actionType);
  if (actionType == 1) sendToFirebase("LIGHT_PRESS", maxPressure);
  else if (actionType == 2) sendToFirebase("HARD_PRESS", maxPressure);
  else sendToFirebase("DOUBLE_SQUEEZE", maxPressure);
  ledActive = true; ledStart = millis();
  Serial.println("30-second alert started.");
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  pinMode(RED_PIN, OUTPUT); pinMode(GREEN_PIN, OUTPUT);
  pinMode(BLUE_PIN, OUTPUT); pinMode(BUZZER_PIN, OUTPUT);
  turnEverythingOff();
  Serial.println("\n================================");
  Serial.println("GripSense TinyML + Rule Fallback + Firebase");
  Serial.println("================================");
  Serial.println("TinyML primary; original pressure rules are fallback.");
  Serial.println("ML window: 150 samples at 20 ms (about 3 seconds).");
  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nWi-Fi connected!");
  Serial.print("IP address: "); Serial.println(WiFi.localIP());
  Serial.println("\nGripSense READY");
}

void loop() {
  int pressure = analogRead(FSR_PIN);
  if (detecting) { Serial.print("Pressure: "); Serial.println(pressure); }

  if (!detecting && !ledActive && pressure > NO_PRESSURE) {
    detecting = true; detectionStart = millis(); maxPressure = pressure;
    consecutiveZeros = 0; doubleSqueezeDetected = false;
    Serial.println("\n>>> Detection started");

    // Keep the original 2-second rule bookkeeping, while collecting the
    // unchanged 3-second TinyML input window. Classification/alerts therefore
    // may start about one second later than in the rule-only sketch.
    unsigned long nextSample = millis();
    bool loggedTwoSeconds = false;
    for (int i = 0; i < ML_SAMPLES; ++i) {
      while ((long)(millis() - nextSample) < 0) delay(1);
      int sample = analogRead(FSR_PIN);
      adcWindow[i] = (uint16_t)sample;
      nextSample += ML_SAMPLE_INTERVAL_MS;
      unsigned long elapsed = millis() - detectionStart;
      if (elapsed < DETECTION_TIME) {
        if (sample > maxPressure) maxPressure = sample;
        if (sample == 0) {
          consecutiveZeros++;
          if (consecutiveZeros >= 2) doubleSqueezeDetected = true;
        } else consecutiveZeros = 0;
      } else if (!loggedTwoSeconds) {
        loggedTwoSeconds = true;
        Serial.println(">>> Original 2-second detection window finished");
        Serial.println("Collecting remaining samples for the 3-second TinyML window.");
      }
    }
    detecting = false;
    Serial.println(">>> Classification window finished");

    int prediction = -1;
    double features[2] = {NAN, NAN};
    bool mlOK = runTinyML(prediction, features);
    Serial.println("--- TinyML window result ---");
    Serial.print("Longest active run fraction: "); Serial.println(features[0], 4);
    Serial.print("Segment 1 mean ADC: "); Serial.println(features[1], 2);

    if (mlOK) {
      Serial.print("Predicted class index: "); Serial.println(prediction);
      const char* label = gs_label_name(prediction);
      Serial.print("Predicted class: "); Serial.println(label ? label : "UNKNOWN");
      // Model IDs: 0 light, 1 hard, 2 double, 3 no press.
      if (prediction == 0) finishAction(1, "TinyML");
      else if (prediction == 1) finishAction(2, "TinyML");
      else if (prediction == 2) finishAction(3, "TinyML");
      else finishAction(0, "TinyML");
    } else {
      Serial.println("TinyML output invalid; using original rule-based fallback.");
      if (doubleSqueezeDetected) finishAction(3, "Rule-based fallback");
      else if (maxPressure <= HARD_PRESS) finishAction(1, "Rule-based fallback");
      else finishAction(2, "Rule-based fallback");
    }
  }

  // Original 30-second alert and buzzer patterns.
  if (ledActive) {
    unsigned long elapsed = millis() - ledStart;
    if (elapsed >= LED_TIME) {
      turnEverythingOff(); ledActive = false; actionType = 0;
      Serial.println("\n>>> Alert finished\n>>> GripSense READY\n");
    } else {
      bool buzzerOn = false;
      if (actionType == 1) buzzerOn = (elapsed % (LIGHT_BEEP_TIME + LIGHT_GAP)) < LIGHT_BEEP_TIME;
      else if (actionType == 2) buzzerOn = (elapsed % (HARD_BEEP_TIME + HARD_GAP)) < HARD_BEEP_TIME;
      else if (actionType == 3) buzzerOn = (elapsed % (DOUBLE_BEEP_TIME + DOUBLE_GAP)) < DOUBLE_BEEP_TIME;
      digitalWrite(BUZZER_PIN, buzzerOn ? HIGH : LOW);
    }
  }
  delay(10);
}
