
#include <Arduino.h>
#include <stdint.h>

extern "C" {
  int gs_predict_from_adc(const uint16_t adc[150]);
  int gs_extract_features(const uint16_t adc[150], double out[2]);
  const char *gs_label_name(int label);
}

const int FSR_PIN = 34;
const int SAMPLES = 150;
const unsigned long SAMPLE_INTERVAL_MS = 20;

uint16_t adcWindow[SAMPLES];

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  delay(1000);

  Serial.println();
  Serial.println("GripSense TinyML ESP32 demo");
  Serial.println("Model: trained depth-2 decision tree");
  Serial.println("Sensor: FSR on GPIO34");
  Serial.println("Window: 150 samples at 20 ms intervals");
}

void loop() {
  unsigned long nextSample = millis();

  for (int i = 0; i < SAMPLES; i++) {
    while ((long)(millis() - nextSample) < 0) {
      delay(1);
    }

    adcWindow[i] = (uint16_t)analogRead(FSR_PIN);
    nextSample += SAMPLE_INTERVAL_MS;
  }

  double features[2];
  gs_extract_features(adcWindow, features);
  int prediction = gs_predict_from_adc(adcWindow);

  Serial.println("--- TinyML window result ---");

  Serial.print("Longest active run fraction: ");
  Serial.println(features[0], 4);

  Serial.print("Segment 1 mean ADC: ");
  Serial.println(features[1], 2);

  Serial.print("Predicted class index: ");
  Serial.println(prediction);

  const char *label = gs_label_name(prediction);
  Serial.print("Predicted class: ");
  Serial.println(label ? label : "INVALID");

  Serial.println("----------------------------");
}
