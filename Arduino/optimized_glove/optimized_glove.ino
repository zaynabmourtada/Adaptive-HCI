// 2022/11/24
// Written by Soham Naik - 11/08/2024
// Modified by Zaynab Mourtada - 12/29/2024
// Glove 1 - Single Light Source for OOK (On-Off Keying) Signal
// Single node: Index finger connected to D3 (PWM-capable pin)
int indexFingerLED = 3;  // PWM-capable pin

float shutterRate = 1000; // Default shutter rate in Hz
float shutterPeriod;      // Shutter rate in microseconds
int pilotGap;             // Delay between each OOK signal
int signalGap;            // Delay between each OOK bit
int pulseWidth;           // Width of each OOK bit LIGHT band
int gapWidth;             // Width of each OOK bit DARK band

// Splits the shutter period into ON time and OFF time
void calculateTiming() {
  shutterPeriod = (1 / shutterRate) * 1e6; // Converting shutter rate to microseconds
  pulseWidth = shutterPeriod * 0.5; // 50% ON
  gapWidth = shutterPeriod * 0.5; // 50% OFF
  pilotGap = shutterPeriod * 10;
  signalGap = shutterPeriod * 1;
}

void setup() {
  pinMode(indexFingerLED, OUTPUT);
  calculateTiming();
}

// Function to emit a Gap between OOK signals
void emitPilotGap() {
  digitalWrite(indexFingerLED, LOW);    // Turn LED OFF
  delayMicroseconds(pilotGap);
}

// Function to create a gap between pulses
void emitGap() {
  for (int i = 0; i < gapWidth; i++) {
    digitalWrite(indexFingerLED, LOW);    // Ensure LED is OFF during gap
    delayMicroseconds(signalGap);
  }
}

// Function to emit a symbol with a specified pattern
void emitSymbol(String binaryPattern) {
  for (char bit : binaryPattern) {
    if (bit == '1') {
      digitalWrite(indexFingerLED, HIGH);   // LED ON for "1" bit
      delayMicroseconds(pulseWidth * 0.1);                 // 10% of pulseWidth for '1'
    } else {
      digitalWrite(indexFingerLED, LOW);    // LED OFF for "0" bit
      delayMicroseconds(gapWidth * 0.9);                // 90% of gapDuration for '0'
  }
  emitGap();                                // Add gap after each symbol
 }
}

void generateOOKSignal() {
  // Emit pilot Gap
  emitPilotGap();
  // Emit symbols as per OOK encoding:
  emitSymbol("10001000");  // User_1 // Soham
  //emitSymbol("11001100");  // User_2 // Deniz
  //emitSymbol("10101010");  // User_3 // Zaynab
  //emitSymbol("11110000");  // User_4 // Alan
}

void loop() {
  generateOOKSignal();  // Continuously generate OOK signal
}
