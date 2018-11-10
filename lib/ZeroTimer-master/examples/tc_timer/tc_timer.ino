#include <ZeroTimer.h>

void setup() {
  SerialUSB.begin(115200);
  TC.startTimer(500000, call);
}

void loop() {
}

void call() {
  SerialUSB.println(micros());
}

