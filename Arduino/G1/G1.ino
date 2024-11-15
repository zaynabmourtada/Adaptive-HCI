// 2022/11/24
//glove 1

// single light source
// pilot + 5 OOK
// GLOVE 1: OOK:
//  F1: 1  00001
//  F2: 2  00010
//  F3: 3  00011
//  F4: 4  00100
//  F5: 5  00101
//  Fw: 16+1  10001

// rolling shutter rate 8000


int L1 = 2;
int L2 = 3;
int L3 = 4;
int L4 = 5;
int L5 = 6;
// wrist node
int Lw = 7;

int widAdj = 10 ;
int gap = 1;
int delayunit = 1;

void setup() {

  pinMode(L1, OUTPUT);
  pinMode(L2, OUTPUT);
  pinMode(L3, OUTPUT);
  pinMode(L4, OUTPUT);
  pinMode(L5, OUTPUT);
  pinMode(Lw, OUTPUT);

}

void indication() {

    //  1 pilot symbol
    //    PMW, the highest brightness

    
    for (int k = 0; k < widAdj; k++) {
      digitalWrite(L1, HIGH);
      digitalWrite(L2, HIGH);
      digitalWrite(L3, HIGH);
      digitalWrite(L4, HIGH);
      digitalWrite(L5, HIGH);
      digitalWrite(Lw, HIGH);
      
      delayMicroseconds(12);

      digitalWrite(L1, LOW);
      digitalWrite(L2, LOW);
      digitalWrite(L3, LOW);
      digitalWrite(L4, LOW);
      digitalWrite(L5, LOW);
      digitalWrite(Lw, LOW);
      
      delayMicroseconds(0);
    }

    for (int k = 0; k < gap; k++) {
      digitalWrite(L1, LOW);
      digitalWrite(L2, LOW);
      digitalWrite(L3, LOW);
      digitalWrite(L4, LOW);
      digitalWrite(L5, LOW);
      digitalWrite(Lw, LOW);
      delayMicroseconds(1);
    }

    //    one symbol width is 120us if widthAdj = 1
    //    on symbol: medium brightness
    //    off symbol: level 0 brightness

    //    symbol 1
    //  F1: 1  00001
    //  F2: 0  00010
    //  F3: 1  00011
    //  F4: 0  00100
    //  F5: 1  00101
    //  Fw: 1  10001

    for (int k = 0; k < widAdj; k++) {
      digitalWrite(L1, HIGH);
      digitalWrite(L2, LOW);
      digitalWrite(L3, HIGH);
      digitalWrite(L4, LOW);
      digitalWrite(L5, HIGH);
      digitalWrite(Lw, HIGH);
      delayMicroseconds(3);

      digitalWrite(L1, LOW);
      digitalWrite(L2, LOW);
      digitalWrite(L3, LOW);
      digitalWrite(L4, LOW);
      digitalWrite(L5, LOW);
      digitalWrite(Lw, LOW);
      delayMicroseconds(9);

    }

    for (int k = 0; k < gap; k++) {
      digitalWrite(L1, LOW);
      digitalWrite(L2, LOW);
      digitalWrite(L3, LOW);
      digitalWrite(L4, LOW);
      digitalWrite(L5, LOW);
      digitalWrite(Lw, LOW);
      delayMicroseconds(1);
    }

    //    symbol 2
    //  F1: 0  00001
    //  F2: 1  00010
    //  F3: 1  00011
    //  F4: 0  00100
    //  F5: 0  00101
    //  Fw: 0  10001

    for (int k = 0; k < widAdj; k++) {
      digitalWrite(L1, LOW);
      digitalWrite(L2, HIGH);
      digitalWrite(L3, HIGH);
      digitalWrite(L4, LOW);
      digitalWrite(L5, LOW);
      digitalWrite(Lw, LOW);
      delayMicroseconds(3);

      digitalWrite(L1, LOW);
      digitalWrite(L2, LOW);
      digitalWrite(L3, LOW);
      digitalWrite(L4, LOW);
      digitalWrite(L5, LOW);
      digitalWrite(Lw, LOW);
      delayMicroseconds(9);
    }

    //    symbol 3
    //  F1: 0  00001
    //  F2: 0  00010
    //  F3: 0  00011
    //  F4: 1  00100
    //  F5: 1  00101
    //  Fw: 0  10001

    for (int k = 0; k < widAdj; k++) {
      digitalWrite(L1, LOW);
      digitalWrite(L2, LOW);
      digitalWrite(L3, LOW);
      digitalWrite(L4, HIGH);
      digitalWrite(L5, HIGH);
      digitalWrite(Lw, LOW);
      delayMicroseconds(3);

      digitalWrite(L1, LOW);
      digitalWrite(L2, LOW);
      digitalWrite(L3, LOW);
      digitalWrite(L4, LOW);
      digitalWrite(L5, LOW);
      digitalWrite(Lw, LOW);
      delayMicroseconds(9);
    }

    for (int k = 0; k < gap; k++) {
      digitalWrite(L1, LOW);
      digitalWrite(L2, LOW);
      digitalWrite(L3, LOW);
      digitalWrite(L4, LOW);
      digitalWrite(L5, LOW);
      digitalWrite(Lw, LOW);
      delayMicroseconds(1);
    }

    //    symbol 4
    //  F1: 0  00001
    //  F2: 0  00010
    //  F3: 0  00011
    //  F4: 0  00100
    //  F5: 0  00101
    //  Fw: 0  10001

    for (int k = 0; k < widAdj; k++) {
      digitalWrite(L1, LOW);
      digitalWrite(L2, LOW);
      digitalWrite(L3, LOW);
      digitalWrite(L4, LOW);
      digitalWrite(L5, LOW);
      digitalWrite(Lw, LOW);
      delayMicroseconds(3);

      digitalWrite(L1, LOW);
      digitalWrite(L2, LOW);
      digitalWrite(L3, LOW);
      digitalWrite(L4, LOW);
      digitalWrite(L5, LOW);
      digitalWrite(Lw, LOW);
      delayMicroseconds(9);
    }

    for (int k = 0; k < gap; k++) {
      digitalWrite(L1, LOW);
      digitalWrite(L2, LOW);
      digitalWrite(L3, LOW);
      digitalWrite(L4, LOW);
      digitalWrite(L5, LOW);
      digitalWrite(Lw, LOW);
      delayMicroseconds(1);
    }

    //    symbol 5
    //  F1: 0  00001
    //  F2: 0  00010
    //  F3: 0  00011
    //  F4: 0  00100
    //  F5: 1  10001

    for (int k = 0; k < widAdj; k++) {
      digitalWrite(L1, LOW);
      digitalWrite(L2, LOW);
      digitalWrite(L3, LOW);
      digitalWrite(L4, LOW);
      digitalWrite(L5, LOW);
      digitalWrite(Lw, HIGH);
      delayMicroseconds(3);

      digitalWrite(L1, LOW);
      digitalWrite(L2, LOW);
      digitalWrite(L3, LOW);
      digitalWrite(L4, LOW);
      digitalWrite(L5, LOW);
      digitalWrite(Lw, LOW);
      delayMicroseconds(9);
    }


}

  void loop() {

    indication();

  }
