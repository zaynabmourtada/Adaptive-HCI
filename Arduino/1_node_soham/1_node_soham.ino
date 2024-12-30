// 2022/11/24
// Written by Soham Naik - 11/08/2024
// Glove 1 - Single Light Source for OOK (On-Off Keying) Signal

// Single node: Index finger connected to D3 (PWM-capable pin)
int indexFingerLED = 3;  // PWM-capable pin

int HIGHwidth = 500;      // HIGH Width of each 1 within the OOK Signal
int LOWwidth = 500;     //  LOW Width of each 0 within the OOK Signal
int OOKgapWidth = 1000;    // Base Delay Between OOK Signals

void setup() {
  pinMode(indexFingerLED, OUTPUT);
}

// Function to create a gap between pulses
void emitGap() {
  digitalWrite(indexFingerLED, LOW);    // Ensure LED is OFF during gap
  delayMicroseconds(OOKgapWidth);
}

// Function to emit a symbol with a specified pattern
void emitSymbol(String binaryPattern) {
  for (char bit : binaryPattern) {
    if (bit == '1') {
      digitalWrite(indexFingerLED, HIGH);   // LED ON for "1" bit
      delayMicroseconds(HIGHwidth);                 // "1" bit on-time
    } else {
      digitalWrite(indexFingerLED, LOW);    // LED OFF for "0" bit
      delayMicroseconds(LOWwidth);                // "0" bit duration
    }
  }
}

void generateOOKSignal() {
  // Emit symbols as per OOK encoding:
  emitSymbol("10");  // F1: 1
  emitGap();
}

void loop() {
  generateOOKSignal();  // Continuously generate OOK signal
}
