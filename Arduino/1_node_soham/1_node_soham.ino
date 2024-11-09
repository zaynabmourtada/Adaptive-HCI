// 2022/11/24
// Written by Soham Naik - 11/08/2024
// Glove 1 - Single Light Source for OOK (On-Off Keying) Signal

// Single node: Index finger connected to D3 (PWM-capable pin)
int indexFingerLED = 3;  // PWM-capable pin

int pulseWidth = 1;     // Adjustable pulse width // Visually Represents the Width of each LIGHT Band
int gapDuration = 100;     // Gap between pulses     // Visually Represents the Width of each DARK Band
int delayUnit = 1;       // Base delay unit in microseconds // Scale both pulseWidth & gapDuration by a constant

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
void emitSymbol(String binaryPattern) {
  for (char bit : binaryPattern) {
    if (bit == '1') {
      digitalWrite(indexFingerLED, HIGH);   // LED ON for "1" bit
      delayMicroseconds(3);                 // "1" bit on-time
    } else {
      digitalWrite(indexFingerLED, LOW);    // LED OFF for "0" bit
      delayMicroseconds(9);                // "0" bit duration
    }
  }
  emitGap();                                // Add gap after each symbol
}

void generateOOKSignal() {
  // Emit pilot pulse
  emitPilotPulse();
  emitGap();

  // Emit symbols as per OOK encoding:
  emitSymbol("00001");  // F1: 1
  //emitSymbol("00010");  // F2: 0
  //emitSymbol("00011");  // F3: 1
  //emitSymbol("00100");  // F4: 0
  //emitSymbol("00101");  // F5: 1
}

void loop() {
  generateOOKSignal();  // Continuously generate OOK signal
}
