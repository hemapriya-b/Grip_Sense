# GripSense – Assistive Communication Through Wearable Technology

GripSense is a low-cost wearable assistive communication system designed to help individuals with speech and communication impairments express their basic needs through simple hand gestures and pressure-based interactions.

The system uses a Force Sensitive Resistor (FSR) integrated into a wearable glove, an ESP32 microcontroller and a TinyML-based classification model to recognize different squeeze patterns. The recognized actions are translated into meaningful alerts that caregivers can view through a dedicated web-based dashboard.

Designed to be affordable and accessible, GripSense combines embedded systems, machine learning and web technologies to provide a simple communication interface without requiring the user to type or speak.

## Key Features

- **Pressure-Based Interaction:** Uses an FSR sensor to detect variations in hand pressure and squeeze patterns.
- **TinyML-Based Classification:** Processes sensor readings using a lightweight machine learning model to identify different user actions.
- **ESP32 Integration:** Reads sensor inputs and processes the recognized actions through an embedded microcontroller.
- **Multiple Communication Patterns:** Distinguishes between light presses, hard presses and double squeezes to represent different user needs.
- **Visual Alerts:** Uses different LED indicators to communicate recognized needs.
- **Caretaker Dashboard:** Provides a web interface for caregivers to monitor patient alerts and view previous interactions.
- **Alert Notifications:** Supports browser notifications and audio alerts to help caregivers notice incoming requests.
- **Patient Profiles:** Includes patient and caretaker profile management through the dashboard.
- **Low-Cost Design:** Developed with an emphasis on affordability, using components that keep the overall prototype cost below ₹1,000.

## How It Works

GripSense follows a sensor-to-alert workflow:

1. The user interacts with the wearable glove by applying pressure or performing a squeeze.
2. The FSR sensor detects the pressure variation and generates corresponding analog readings.
3. The ESP32 reads the sensor values and processes the input using the trained TinyML classification model.
4. The recognized action is mapped to a specific user need.
5. The corresponding visual indicator is activated, and the event is communicated to the caretaker dashboard.
6. Caregivers can view the latest alert and access the interaction history through the web interface.

## Gesture and Alert Mapping

The system recognizes three primary interaction patterns:

- **Light Press – Hungry:** Indicates that the user needs food. Represented using a green LED.
- **Double Squeeze – Washroom:** Indicates a washroom requirement. Represented using a blue LED.
- **Hard Press – Pain:** Indicates discomfort or pain. Represented using a red LED.

When no pressure is detected, the system remains in its idle state.

## Machine Learning and TinyML

GripSense incorporates a lightweight machine learning model to distinguish between different pressure-based interactions.

A dataset was developed by collecting sensor readings across multiple interaction patterns. The collected data was used to train and evaluate classification models, with the selected lightweight decision tree model designed for efficient embedded deployment.

The implementation demonstrates the use of TinyML to bring machine learning capabilities to resource-constrained microcontrollers.

## Hardware Components

- ESP32 Microcontroller
- Force Sensitive Resistor (FSR)
- 10kΩ Resistor
- Wearable Glove
- LED Indicators
- Connecting Wires and Supporting Circuit Components

The FSR is connected through a voltage-divider circuit to the ESP32 analog input, allowing pressure variations to be converted into readable sensor values.

## Technology Stack

**Embedded Systems:** ESP32, Arduino

**Sensor Integration:** Force Sensitive Resistor (FSR), Analog-to-Digital Conversion

**Machine Learning:** Python, Scikit-learn, TinyML, Decision Tree Classification

**Web Technologies:** HTML5, CSS3, JavaScript

**Backend and Data Management:** Firebase Realtime Database, Firebase Authentication

**Development Tools:** Arduino IDE, Python, Git, GitHub

## Caretaker Dashboard

GripSense includes a web-based caretaker dashboard designed to provide visibility into patient interactions.

The dashboard supports:

- Caretaker registration and login.
- Patient profile management.
- Real-time monitoring of incoming patient alerts.
- Display of the latest recognized need.
- Historical alert tracking.
- Browser notifications and audio-based alerts.
- Responsive layouts for mobile and desktop devices.

The ESP32 communicates recognized interaction events to Firebase Realtime Database, which allows the dashboard to retrieve and display the corresponding alerts.

## Getting Started – Caretaker Dashboard

To run the dashboard locally, Python 3 is required.

1. Clone the repository:

```bash
git clone https://github.com/hemapriya-b/Grip_Sense.git
```

2. Navigate to the project directory.

3. Copy `firebase-config.example.js` to `firebase-config.js` and configure the Firebase Web App settings using your Firebase project credentials.

4. Start a local development server:

```bash
python3 -m http.server 8000
```

5. Open the following address in your browser:

```text
http://localhost:8000
```

The application requires a local server to load its JavaScript ES modules correctly.

**Note:** The Firebase configuration file must contain only the standard Firebase Web App configuration. Never expose service account credentials or private keys.

## Project Objective

GripSense aims to make everyday communication more accessible for individuals who face difficulties expressing their basic needs verbally. By combining wearable sensing, embedded machine learning and a caregiver monitoring interface, the project explores an affordable assistive technology solution that translates simple physical interactions into meaningful communication.

---

**GripSense** | Wearable Assistive Communication System
