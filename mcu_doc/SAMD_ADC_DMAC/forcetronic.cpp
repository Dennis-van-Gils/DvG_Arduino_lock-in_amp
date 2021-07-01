
/*This code is from a tutorial on the ForceTronics YouTube Channel that talks about speeding up the sample rate on Arduino boards 
 * that use the SAMD21 microcontroller like the Arduino Zero or MKR series. This code is free and clear for other to use and modify 
 * at their own risk. 
 
 SOURCE: https://forcetronic.blogspot.com/2018/06/speeding-up-adc-on-arduino-samd21.html
 */
#include <SPI.h>
#include <SD.h>

const int16_t dSize = 1024; //used to set number of samples
const byte chipSelect = 38; //used for SPI chip select pin
const byte gClk = 3; //used to define which generic clock we will use for ADC
const byte intPri = 0; //used to set interrupt priority for ADC
const int cDiv = 1; //divide factor for generic clock
const float period = 3.3334; //period of 300k sample rate
String wFile = "ADC_DATA"; //used as file name to store wind and GPS data
volatile int aDCVal[dSize]; //array to hold ADC samples
volatile int count = 0; //tracks how many samples we have collected
bool done = false; //tracks when done writing data to SD card

void setup() {
  portSetup(); //setup the ports or pin to make ADC measurement
  genericClockSetup(gClk,cDiv); //setup generic clock and routed it to ADC
  aDCSetup(); //this function set up registers for ADC, input argument sets ADC reference
  setUpInterrupt(intPri); //sets up interrupt for ADC and argument assigns priority
  aDCSWTrigger(); //trigger ADC to start free run mode
}

void loop() {
  
  if(count==(dSize-1) and !done) { //if done reading and they have not been written to SD card yet
    removeDCOffset(aDCVal, dSize, 8); //this function removes DC offset if you are measuring an AC signal
    SD.begin(chipSelect); //start SD card library
    File myFile = SD.open((wFile + ".csv"), FILE_WRITE); //open file to write data to CSV file
    if (myFile) {
      float sTime = 0;
      for (int y = 0; y < dSize; y++) {
        myFile.print(sTime,5); //write each reading to SD card as string
        myFile.print(",");
        myFile.println(String(aDCVal[y])+","); //write each reading to SD card as string
        sTime = sTime + period; //update signal period info
      }
    }
    myFile.close(); //close file
    done = true; //we are done 
  }
}

//function for configuring ports or pins, note that this will not use the same pin numbering scheme as Arduino
void portSetup() {
  // Input pin for ADC Arduino A0/PA02
  REG_PORT_DIRCLR1 = PORT_PA02;

  // Enable multiplexing on PA02_AIN0 PA03/ADC_VREFA
  PORT->Group[0].PINCFG[2].bit.PMUXEN = 1;
  PORT->Group[0].PINCFG[3].bit.PMUXEN = 1;
  PORT->Group[0].PMUX[1].reg = PORT_PMUX_PMUXE_B | PORT_PMUX_PMUXO_B;
}

//this function sets up the generic clock that will be used for the ADC unit
//by default it uses the 48M system clock, input arguments set divide factor for generic clock and choose which generic clock
//Note unless you understand how the clock system works use clock 3. clocks 5 and up can brick the microcontroller based on how Arduino configures things
void genericClockSetup(int clk, int dFactor) {
  // Enable the APBC clock for the ADC
  REG_PM_APBCMASK |= PM_APBCMASK_ADC;
  
  //This allows you to setup a div factor for the selected clock certain clocks allow certain division factors: Generic clock generators 3 - 8 8 division factor bits - DIV[7:0]
  GCLK->GENDIV.reg |= GCLK_GENDIV_ID(clk)| GCLK_GENDIV_DIV(dFactor);
  while (GCLK->STATUS.reg & GCLK_STATUS_SYNCBUSY);  

  //configure the generator of the generic clock with 48MHz clock
  GCLK->GENCTRL.reg |= GCLK_GENCTRL_GENEN | GCLK_GENCTRL_SRC_DFLL48M | GCLK_GENCTRL_ID(clk); // GCLK_GENCTRL_DIVSEL don't need this, it makes divide based on power of two
  while (GCLK->STATUS.reg & GCLK_STATUS_SYNCBUSY);
  
  //enable clock, set gen clock number, and ID to where the clock goes (30 is ADC)
  GCLK->CLKCTRL.reg |= GCLK_CLKCTRL_CLKEN | GCLK_CLKCTRL_GEN(clk) | GCLK_CLKCTRL_ID(30);
  while (GCLK->STATUS.bit.SYNCBUSY);
}

/*
ADC_CTRLB_PRESCALER_DIV4_Val    0x0u  
ADC_CTRLB_PRESCALER_DIV8_Val    0x1u   
ADC_CTRLB_PRESCALER_DIV16_Val   0x2u   
ADC_CTRLB_PRESCALER_DIV32_Val   0x3u   
ADC_CTRLB_PRESCALER_DIV64_Val   0x4u   
ADC_CTRLB_PRESCALER_DIV128_Val  0x5u   
ADC_CTRLB_PRESCALER_DIV256_Val  0x6u   
ADC_CTRLB_PRESCALER_DIV512_Val  0x7u   
--> 8 bit ADC measurement takes 5 clock cycles, 10 bit ADC measurement takes 6 clock cycles
--> Using 48MHz system clock with division factor of 1
--> Using ADC division factor of 32
--> Sample rate = 48M / (5 x 32) = 300 KSPS
This function sets up the ADC, including setting resolution and ADC sample rate
*/
void aDCSetup() {
  // Select reference
  REG_ADC_REFCTRL = ADC_REFCTRL_REFSEL_INTVCC1; //set vref for ADC to VCC

  // Average control 1 sample, no right-shift
  REG_ADC_AVGCTRL |= ADC_AVGCTRL_SAMPLENUM_1;

  // Sampling time, no extra sampling half clock-cycles
  REG_ADC_SAMPCTRL = ADC_SAMPCTRL_SAMPLEN(0);
  
  // Input control and input scan
  REG_ADC_INPUTCTRL |= ADC_INPUTCTRL_GAIN_1X | ADC_INPUTCTRL_MUXNEG_GND | ADC_INPUTCTRL_MUXPOS_PIN0;
  // Wait for synchronization
  while (REG_ADC_STATUS & ADC_STATUS_SYNCBUSY);

  ADC->CTRLB.reg |= ADC_CTRLB_RESSEL_8BIT | ADC_CTRLB_PRESCALER_DIV32 | ADC_CTRLB_FREERUN; //This is where you set the divide factor, note that the divide call has no effect until you change Arduino wire.c
  //Wait for synchronization
  while (REG_ADC_STATUS & ADC_STATUS_SYNCBUSY);

  ADC->WINCTRL.reg = ADC_WINCTRL_WINMODE_DISABLE; // Disable window monitor mode
  while(ADC->STATUS.bit.SYNCBUSY);

  ADC->EVCTRL.reg |= ADC_EVCTRL_STARTEI; //start ADC when event occurs
  while (ADC->STATUS.bit.SYNCBUSY);

  ADC->CTRLA.reg |= ADC_CTRLA_ENABLE; //set ADC to run in standby
  while (ADC->STATUS.bit.SYNCBUSY);
}

//This function sets up an ADC interrupt that is triggered 
//when an ADC value is out of range of the window
//input argument is priority of interrupt (0 is highest priority)
void setUpInterrupt(byte priority) {
  
  ADC->INTENSET.reg |= ADC_INTENSET_RESRDY; // enable ADC ready interrupt
  while (ADC->STATUS.bit.SYNCBUSY);

  NVIC_EnableIRQ(ADC_IRQn); // enable ADC interrupts
  NVIC_SetPriority(ADC_IRQn, priority); //set priority of the interrupt
}

//software trigger to start ADC in free run
//in future could use this to set various ADC triggers
void aDCSWTrigger() {
  ADC->SWTRIG.reg |= ADC_SWTRIG_START;
}

//This ISR is called each time ADC makes a reading
void ADC_Handler() {
    if(count<1023) {
      aDCVal[count] = REG_ADC_RESULT;
      count++;
    }
    ADC->INTFLAG.reg = ADC_INTENSET_RESRDY; //Need to reset interrupt
}

//This function takes out DC offset of AC signal, it assumes that the offset brings signal to zero volts
//input arguments: array with measured points and bits of measurement
void removeDCOffset(volatile int aDC[], int aSize, int bits) {
  int aSteps = pow(2,bits)/2; //get number of levels in ADC measurement and cut it in half
  for(int i=0; i<aSize; i++) {
    aDC[i] = aDC[i] - aSteps; //take out offset
  }
}
