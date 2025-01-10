// 2022/11/24
// Written by Soham Naik - 11/08/2024
// Modified by Zaynab Mourtada - 12/29/2024
// Glove 1 - Single Light Source for OOK (On-Off Keying) Signal
// Single node: Index finger connected to D3 (PWM-capable pin)

int indexFingerLED = 3;  // PWM-capable pin

float shutterRate = 1000.0; // Default shutter rate in Hz
float shutterPeriod;        // Shutter period in microseconds
int pilotGap;               // Delay between each OOK signal
int pulseWidth;             // Width of each OOK bit LIGHT band
int gapWidth;               // Width of each OOK bit DARK band

// Splits the shutter period into ON time and OFF time
void calculateTiming() {
  shutterPeriod = (1.0 / shutterRate) * 1e6; // Converting shutter rate to microseconds
  pulseWidth = shutterPeriod * 1.0;         // 50% ON
  gapWidth = shutterPeriod * 1.0;           // 50% OFF
  pilotGap = shutterPeriod * 7;
}

// Function to emit a gap between OOK signals
void emitPilotGap() {
  digitalWrite(indexFingerLED, LOW);    // Turn LED OFF
  delayMicroseconds(pilotGap);
}

// Function to emit a symbol with a specified pattern
void emitSymbol(const char* binaryPattern) {
  for (int i = 0; binaryPattern[i] != '\0'; i++) {
    if (binaryPattern[i] == '1') {
      digitalWrite(indexFingerLED, HIGH);  // LED ON for "1" bit
      delayMicroseconds(pulseWidth);
    } else {
      digitalWrite(indexFingerLED, LOW);   // LED OFF for "0" bit
      delayMicroseconds(gapWidth);
    }
  }
}

// Function to generate the OOK signal
void generateOOKSignal() {
  //emitPilotGap();                     // Emit pilot gap
  //emitSymbol("11110000");        // User_1
  emitSymbol("10101010");          // User_2

void setup() {
  pinMode(indexFingerLED, OUTPUT);
  calculateTiming();
}

void loop() {
  generateOOKSignal();  // Continuously generate OOK signal
}