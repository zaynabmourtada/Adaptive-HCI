// 2022/11/24
// Written by Soham Naik - 11/08/2024
// Modified by Zaynab Mourtada - 12/29/2024
// Glove 1 - Two Light Sources for OOK Signal
// Toggles LED_1, LED_2 and LED_3 simultaneously

int LED_1 = 7; // Thumb
int LED_2 = 4; // Pinky
int LED_3 = 3; // Index

float shutterRate = 6000.0;  
unsigned long shutterPeriod;

char USER_1[] = "10101010";
char USER_2[] = "11110000";
char USER_3[] = "11111111";

void updateLED(int pin, int& idx, const char* pattern) {
  digitalWrite(pin, (pattern[idx] == '1') ? HIGH : LOW);
  idx = (idx + 1) % strlen(pattern);
}

void setup() {
  pinMode(LED_1, OUTPUT);
  pinMode(LED_2, OUTPUT);
  pinMode(LED_3, OUTPUT);
  // Convert rate in Hz to a period in microseconds
  shutterPeriod = (unsigned long)((1.0 / shutterRate) * 1e6);
}

void loop() {
  static int i1 = 0, i2 = 0, i3 = 0;
  unsigned long startTime = micros();

  // Update all LED's
  updateLED(LED_1, i1, USER_1);
  //updateLED(LED_2, i2, USER_2);
  //updateLED(LED_3, i3, USER_3);

  // Measure how long the updates took
  unsigned long endTime = micros();
  unsigned long elapsed = endTime - startTime;

  // If there's still time left in this cycle, wait out the remainder
  if (elapsed < shutterPeriod) {
    delayMicroseconds(shutterPeriod - elapsed);
  }
}
