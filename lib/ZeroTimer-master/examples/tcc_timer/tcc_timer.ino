#include <ZeroTimer.h>

void setup() {
  SerialUSB.begin(115200);
  TCC.startTimer(1000000, call);
}

void loop() {
}

void call() {
  SerialUSB.println(micros());
}