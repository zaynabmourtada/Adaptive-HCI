// 2022/11/24
// Written by Soham Naik - 11/08/2024
// Glove 1 - Single Light Source for OOK (On-Off Keying) Signal

// Single node: Index finger connected to D3 (PWM-capable pin)
int indexFingerLED = 3;  // PWM-capable pin

int pulseWidth = 100;     // Adjustable pulse width
int gapDuration = 100;     // Gap between pulses
int delayUnit = 100;       // Base delay unit in microseconds

void setup() {
  pinMode(indexFingerLED, OUTPUT);
}

// Function to emit a pilot pulse at maximum brightness
void emitPilotPulse() {
  for (int i = 0; i < pulseWidth; i++) {
    digitalWrite(indexFingerLED, HIGH);   // Turn LED ON (full brightness)
    delayMicroseconds(12);                // Pilot pulse duration
    digitalWrite(indexFingerLED, LOW);    // Turn LED OFF
    delayMicroseconds(0);                 // Minimal delay before next pulse
  }
}

// Function to create a gap between pulses
void emitGap() {
  for (int i = 0; i < gapDuration; i++) {
    digitalWrite(indexFingerLED, LOW);    // Ensure LED is OFF during gap
    delayMicroseconds(delayUnit);
  }
}

// Function to emit a symbol with a specified pattern
void emitSymbol(int onTime, int offTime) {
  for (int i = 0; i < pulseWidth; i++) {
    digitalWrite(indexFingerLED, HIGH);   // LED ON for onTime
    delayMicroseconds(onTime);
    digitalWrite(indexFingerLED, LOW);    // LED OFF for offTime
    delayMicroseconds(offTime);
  }
  emitGap();                              // Add gap after symbol
}

void generateOOKSignal() {
  // Emit pilot pulse
  emitPilotPulse();
  emitGap();

  // Emit symbols as per OOK encoding:
  // F1: 1  00001
  emitSymbol(3, 9);

  // F2: 0  00010
  emitSymbol(3, 9);

  // F3: 1  00011
  emitSymbol(3, 9);

  // F4: 0  00100
  emitSymbol(3, 9);

  // F5: 1  00101
  emitSymbol(3, 9);
}

void loop() {
  generateOOKSignal();  // Continuously generate OOK signal
}
