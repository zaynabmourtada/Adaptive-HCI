// 2022/11/24
// Written by Soham Naik - 11/08/2024
// Modified by Zaynab Mourtada - 12/29/2024
// Glove 1 - Two Light Sources for OOK Signal
// Example of toggling LED_1 and LED_2 simultaneously

int LED_1 = 7; // Thumb
int LED_2 = 4; // Pinky

float shutterRate = 6000.0;  
float shutterPeriod;

char USER_1[] = "10101010";
char USER_2[] = "11110000";

void setup() {
  pinMode(LED_1, OUTPUT);
  pinMode(LED_2, OUTPUT);
  shutterPeriod = (1.0 / shutterRate) * 1e6;
}

void updateLED(int pin, int& index, const char* pattern) {
  digitalWrite(pin, (pattern[index] == '1') ? HIGH : LOW);
  index = (index + 1) % strlen(pattern);
}

void loop() {
  static int i1 = 0, i2 = 0;

  updateLED(LED_1, i1, USER_1);
  updateLED(LED_2, i2, USER_2);

  delayMicroseconds(shutterPeriod);
}
