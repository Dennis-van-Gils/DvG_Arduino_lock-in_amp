#include <ZeroTimer.h>

const int led = 13;

volatile bool state = LOW;

void setup() {
  pinMode(led, OUTPUT);
  SerialUSB.begin(115200);
  TC.startTimer(700000, blink);
  TCC.startTimer(1000000, call);
}

void loop() {
}

void blink() {
  digitalWrite(led, state);
  state = !state;
}

void call() {
  SerialUSB.println(micros());
}
