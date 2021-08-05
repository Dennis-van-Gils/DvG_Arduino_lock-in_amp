# FlashStorage_SAMD library for Arduino

[![arduino-library-badge](https://www.ardu-badge.com/badge/FlashStorage_SAMD.svg?)](https://www.ardu-badge.com/FlashStorage_SAMD)
[![GitHub release](https://img.shields.io/github/release/khoih-prog/FlashStorage_SAMD.svg)](https://github.com/khoih-prog/FlashStorage_SAMD/releases)
[![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/khoih-prog/FlashStorage_SAMD/blob/master/LICENSE)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](#Contributing)
[![GitHub issues](https://img.shields.io/github/issues/khoih-prog/FlashStorage_SAMD.svg)](http://github.com/khoih-prog/FlashStorage_SAMD/issues)

---
---

## Table of Contents

* [Why do we need this FlashStorage_SAMD library](#why-do-we-need-this-flashstorage_samd-library)
  * [Features](#features)
  * [Currently supported Boards](#currently-supported-boards)
* [Changelog](#changelog)
  * [Releases v1.1.0](#releases-v110)
  * [Releases v1.0.0](#releases-v100)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
  * [Use Arduino Library Manager](#use-arduino-library-manager)
  * [Manual Install](#manual-install)
  * [VS Code & PlatformIO](#vs-code--platformio)
* [Packages' Patches](#packages-patches)
  * [1. For Arduino SAMD boards](#1-for-arduino-samd-boards)
      * [For core version v1.8.10+](#for-core-version-v1810)
      * [For core version v1.8.9-](#for-core-version-v189-)
  * [2. For Adafruit SAMD boards](#2-for-adafruit-samd-boards)
  * [3. For Seeeduino SAMD boards](#3-for-seeeduino-samd-boards)
* [Limited number of writes](#limited-number-of-writes)
* [Usage](#usage)
  * [Using FlashStorage objects directly](#using-flashstorage-objects-directly)
  * [Using the alternative EEPROM-like API](#using-the-alternative-eeprom-like-api)
* [Examples](#examples)
  * [  1. EEPROM_Clear](examples/EEPROM_Clear)
  * [  2. EEPROM_CRC](examples/EEPROM_CRC)
  * [  3. EEPROM_get](examples/EEPROM_get)
  * [  4. EEPROM_iteration](examples/EEPROM_iteration)
  * [  5. EEPROM_put](examples/EEPROM_put)
  * [  6. EEPROM_read](examples/EEPROM_read)
  * [  7. EEPROM_update](examples/EEPROM_update)
  * [  8. EEPROM_write](examples/EEPROM_write)
  * [  9. **EmulateEEPROM**](examples/EmulateEEPROM)
  * [ 10. **FlashStoreAndRetrieve**](examples/FlashStoreAndRetrieve)
  * [ 11. **StoreNameAndSurname**](examples/StoreNameAndSurname)
* [Examples from other libraries](#examples-from-other-libraries)
  * [  1. Library BlynkEthernet_WM](#1-library-blynkethernet_wm)
  * [  2. Library Blynk_Esp8266AT_WM](#2-library-blynk_esp8266at_wm)
  * [  3. Library Blynk_WiFiNINA_WM](#3-library-blynk_wifinina_wm)
  * [  4. Library ESP_AT_WiFiManager](#4-library-esp_at_wifimanager)
  * [  5. Library ESP_AT_WM_Lite](#5-library-esp_at_wm_lite)
  * [  6. Library WiFiManager_NINA_Lite](#6-library-wifimanager_nina_lite)
  * [  7. Library DoubleResetDetector_Generic](#7-library-doubleresetdetector_generic)
  * [  8. Library MultiResetDetector_Generic](#8-library-multiresetdetector_generic)
* [Example StoreNameAndSurname](#example-storenameandsurname)
* [Debug Terminal Output Samples](#debug-terminal-output-samples)
  * [1. W5500_Blynk on Adafruit SAMD51 ITSYBITSY_M4 using W5500 Ethernet shield](#1-w5500_blynk-on-adafruit-samd51-itsybitsy_m4-using-w5500-ethernet-shield)
  * [2. StoreNameAndSurname on SAMD21 SEEED_XIAO_M0](#2-storenameandsurname-on-samd21-seeed_xiao_m0)
    * [2.1. No data, input then save](#21-no-data--input-then-save)
    * [2.2. Data retention after reset](#22-data-retention-after-reset)
  * [3. EEPROM_Clear on SAMD21 SEEED_XIAO_M0](#3-eeprom_clear-on-samd21-seeed_xiao_m0)
  * [4. EEPROM_get on SAMD21 SEEED_XIAO_M0](#4-eeprom_get-on-samd21-seeed_xiao_m0)
* [FAQ](#faq)
  * [Can I use a single object to store more stuff?](#can-i-use-a-single-object-to-store-more-stuff)
  * [The content of the FlashStorage is erased each time a new sketch is uploaded?](#the-content-of-the-flashstorage-is-erased-each-time-a-new-sketch-is-uploaded)
  * [Do you recommend to use FLASH instead of EEPROM?](#do-you-recommend-to-use-flash-instead-of-eeprom)
* [Troubleshooting](#troubleshooting)
* [Releases](#releases)
* [Issues](#issues)
* [TO DO](#to-do)
* [DONE](#done)
* [Contributions and Thanks](#contributions-and-thanks)
* [Contributing](#contributing)
* [License](#license)
* [Copyright](#copyright)

---
---

### Why do we need this [FlashStorage_SAMD library](https://github.com/khoih-prog/FlashStorage_SAMD)

### Features

The FlashStorage_SAMD library, based on and modified from [Cristian Maglie's FlashStorage](https://github.com/cmaglie/FlashStorage), provides a convenient way to store and retrieve user's data using emulated-EEPROM, from the non-volatile flash memory of SAMD21/SAMD51 boards.

The flash memory, generally used to store the firmware code, can also be used to store / retrieve more user's data and faster than from EEPROM. Thanks to the **buffered data writing and reading**, the flash access time is greatly reduced to **increase the life of the flash**.

This [**FlashStorage_SAMD library**](https://github.com/khoih-prog/FlashStorage_SAMD) is based on and modified from:


### Currently supported Boards

  - **Arduino SAMD21 (ZERO, MKR, NANO_33_IOT, etc.)**.
  - **Adafruit SAM21 (Itsy-Bitsy M0, Metro M0, Feather M0, Gemma M0, etc.)**.
  - **Adafruit SAM51 (Itsy-Bitsy M4, Metro M4, Grand Central M4, Feather M4 Express, etc.)**.
  - **Seeeduino SAMD21/SAMD51 boards (SEEED_WIO_TERMINAL, SEEED_FEMTO_M0, SEEED_XIAO_M0, Wio_Lite_MG126, WIO_GPS_BOARD, SEEEDUINO_ZERO, SEEEDUINO_LORAWAN, SEEED_GROVE_UI_WIRELESS, etc.)** 
  
---
---

## Changelog

### Releases v1.1.0

1. Add supports to `EEPROM.put()` and `EEPROM.get()` for efficiently writing and reading the whole object. 
2. Add `_commitASAP` flag to permit reduce the number of calls to `EEPROM.commit()` to extend the flash life.
3. Add many modified examples from [Arduino EEPROM library](https://www.arduino.cc/en/Reference/EEPROM)
4. Fix bug.

### Releases v1.0.0

1. Add support to SAMD51 family such as 

  - Adafruit SAMD51 (M4): Metro M4, Grand Central M4, ItsyBitsy M4, Feather M4 Express, Trellis M4, Metro M4 AirLift Lite, MONSTER M4SK Express, Hallowing M4, etc.
  - Seeeduino: Wio Terminal, Grove UI WirelessAdafruit Itsy-Bitsy M4, etc.

---
---
 
## Prerequisites

 1. [`Arduino IDE v1.8.13+` for Arduino](https://www.arduino.cc/en/Main/Software)
 2. [`Arduino SAMD core v1.8.11+`](https://www.arduino.cc/en/Guide/ArduinoM0) for SAMD ARM Cortex-M0+ boards.
 3. [`Adafruit SAMD core v1.6.4+`](https://www.adafruit.com/) for SAMD ARM Cortex-M0+ and M4 boards (Nano 33 IoT, etc.).
 4. [`Seeeduino SAMD core 1.8.1+`](https://www.seeedstudio.com/) for SAMD21/SAMD51 boards (XIAO M0, Wio Terminal, etc.).

---

## Installation

### Use Arduino Library Manager

The best and easiest way is to use `Arduino Library Manager`. Search for [**FlashStorage_SAMD**](https://github.com/khoih-prog/FlashStorage_SAMD), then select / install the latest version.
You can also use this link [![arduino-library-badge](https://www.ardu-badge.com/badge/FlashStorage_SAMD.svg?)](https://www.ardu-badge.com/FlashStorage_SAMD) for more detailed instructions.

### Manual Install

Another way to install is to:

1. Navigate to [FlashStorage_SAMD](https://github.com/khoih-prog/FlashStorage_SAMD) page.
2. Download the latest release `FlashStorage_SAMD-master.zip`.
3. Extract the zip file to `FlashStorage_SAMD-master` directory 
4. Copy whole 
  - `FlashStorage_SAMD-master` folder to Arduino libraries' directory such as `~/Arduino/libraries/`.

### VS Code & PlatformIO

1. Install [VS Code](https://code.visualstudio.com/)
2. Install [PlatformIO](https://platformio.org/platformio-ide)
3. Install [**FlashStorage_SAMD** library](https://platformio.org/lib/show/11242/FlashStorage_SAMD) by using [Library Manager](https://platformio.org/lib/show/11242/FlashStorage_SAMD/installation). Search for **FlashStorage_SAMD** in [Platform.io Author's Libraries](https://platformio.org/lib/search?query=author:%22Khoi%20Hoang%22)
4. Use included [platformio.ini](platformio/platformio.ini) file from examples to ensure that all dependent libraries will installed automatically. Please visit documentation for the other options and examples at [Project Configuration File](https://docs.platformio.org/page/projectconf.html)

---
---

### Packages' Patches

#### 1. For Arduino SAMD boards
 
 ***To be able to compile without error and automatically detect and display BOARD_NAME on Arduino SAMD (Nano-33-IoT, etc) boards***, you have to copy the whole [Arduino SAMD cores 1.8.11](Packages_Patches/arduino/hardware/samd/1.8.11) directory into Arduino SAMD directory (~/.arduino15/packages/arduino/hardware/samd/1.8.11).
 
#### For core version v1.8.10+

Supposing the Arduino SAMD version is 1.8.11. Now only one file must be copied into the directory:

- `~/.arduino15/packages/arduino/hardware/samd/1.8.11/platform.txt`

Whenever a new version is installed, remember to copy this files into the new version directory. For example, new version is x.yy.zz

This file must be copied into the directory:

- `~/.arduino15/packages/arduino/hardware/samd/x.yy.zz/platform.txt`
 
#### For core version v1.8.9-

Supposing the Arduino SAMD version is 1.8.9. These files must be copied into the directory:

- `~/.arduino15/packages/arduino/hardware/samd/1.8.9/platform.txt`
- ***`~/.arduino15/packages/arduino/hardware/samd/1.8.9/cores/arduino/Arduino.h`***

Whenever a new version is installed, remember to copy these files into the new version directory. For example, new version is x.yy.z

These files must be copied into the directory:

- `~/.arduino15/packages/arduino/hardware/samd/x.yy.z/platform.txt`
- ***`~/.arduino15/packages/arduino/hardware/samd/x.yy.z/cores/arduino/Arduino.h`***
 
 This is mandatory to fix the ***notorious Arduino SAMD compiler error***. See [Improve Arduino compatibility with the STL (min and max macro)](https://github.com/arduino/ArduinoCore-samd/pull/399)
 
```
 ...\arm-none-eabi\include\c++\7.2.1\bits\stl_algobase.h:243:56: error: macro "min" passed 3 arguments, but takes just 2
     min(const _Tp& __a, const _Tp& __b, _Compare __comp)
```

Whenever the above-mentioned compiler error issue is fixed with the new Arduino SAMD release, you don't need to copy the `Arduino.h` file anymore.

#### 2. For Adafruit SAMD boards
 
 ***To be able to automatically detect and display BOARD_NAME on Adafruit SAMD (Itsy-Bitsy M4, etc) boards***, you have to copy the file [Adafruit SAMD platform.txt](Packages_Patches/adafruit/hardware/samd/1.6.4) into Adafruit samd directory (~/.arduino15/packages/adafruit/hardware/samd/1.6.4). 

Supposing the Adafruit SAMD core version is 1.6.4. This file must be copied into the directory:

- `~/.arduino15/packages/adafruit/hardware/samd/1.6.4/platform.txt`

Whenever a new version is installed, remember to copy this file into the new version directory. For example, new version is x.yy.zz
This file must be copied into the directory:

- `~/.arduino15/packages/adafruit/hardware/samd/x.yy.zz/platform.txt`

#### 3. For Seeeduino SAMD boards
 
 ***To be able to automatically detect and display BOARD_NAME on Seeeduino SAMD (XIAO M0, Wio Terminal, etc) boards***, you have to copy the file [Seeeduino SAMD platform.txt](Packages_Patches/Seeeduino/hardware/samd/1.8.1) into Adafruit samd directory (~/.arduino15/packages/Seeeduino/hardware/samd/1.8.1). 

Supposing the Seeeduino SAMD core version is 1.8.1. This file must be copied into the directory:

- `~/.arduino15/packages/Seeeduino/hardware/samd/1.8.1/platform.txt`

Whenever a new version is installed, remember to copy this file into the new version directory. For example, new version is x.yy.zz
This file must be copied into the directory:

- `~/.arduino15/packages/Seeeduino/hardware/samd/x.yy.zz/platform.txt`


---
---

### Limited number of writes

The flash memory has a limited amount of write cycles. Typical flash memories can perform about 10000 writes cycles to the same flash block before starting to "wear out" and begin to lose the ability to retain data.

So **BEWARE: IMPROPER USE OF THIS LIBRARY CAN QUICKLY AND PERMANENTLY DESTROY THE FLASH MEMORY OF YOUR MICRO**, in particular you should avoid to call the `write()` function too often and make sure that in the entire life of the micro the number of calls to `write` stay well below the above limit of 10000 (it's a good rule-of-thumb to keep that number in mind even if the
manufacturer of the micro guarantees a bigger number of cycles).

The same caution must be taken if you're using the EEPROM API emulation (see below) with the `EEPROM.commit()` function.

---
---

## Usage

### Using FlashStorage objects directly

First of all you must declare a global `FlashStorage` object for each data you intend to store in the flash memory.

For example if you want to store the age of a person, you must declare an `age_storage` like this:

```c++
FlashStorage(age_storage, int);
```

this instruction means "create a `FlashStorage` to store an `int` variable and call it `age_storage`". Now you can use `age_storage` as a place to safely store an integer:

```c++
void readAndStoreUserAge() 
{
  Serial.println("Please enter your age:");
  String age = Serial.readStringUntil('\n');

  age_storage.write(age.toInt());  // <-- save the age
}
```

after a reset of the microcontroller to retrieve the stored age you can use:

```c++
int user_age = age_storage.read();
```

### Using the alternative EEPROM-like API

Include `FlashAsEEPROM_SAMD.h` to get an EEPROM emulation with the internal flash memory.

See [EmulateEEPROM](examples/EmulateEEPROM) sketch for an example.

The API is very similar to the well-known [Arduino EEPROM library API](https://www.arduino.cc/en/Reference/EEPROM), but with 4 additional functions:

* `bool isValid()` returns `true` if data in the emulated-EEPROM is valid (the data written to flash at least once by `EEPROM.commit()` or `EEPROM.put()`). Otherwise emulated-EEPROM data is "undefined" and the function returns `false`.
* `void commit()` store the EEPROM data in flash. Use this with care: Every call writes the complete emulated-EEPROM data to flash. This will reduce the remaining flash-write-cycles. Don't call this method in a loop or [you will kill your flash soon](#limited-number-of-writes).
* `void setCommitASAP(bool value = true)` to set or clear the `_commitASAP` private variable (default is `true` to be safe). If _commitASAP is false, the call to `EEPROM.put()` won't force the `EEPROM.commit()` to extend the flash life. You'll have to remember to call `EEPROM.commit()` manually to save the emulated-EEPROM data into flash or data will be lost.
* `bool getCommitASAP()` to return the current value of `_commitASAP`.


---
---

### Examples

 1. [EEPROM_Clear](examples/EEPROM_Clear)
 2. [EEPROM_CRC](examples/EEPROM_CRC)
 3. [EEPROM_get](examples/EEPROM_get)
 4. [EEPROM_iteration](examples/EEPROM_iteration)
 5. [EEPROM_put](examples/EEPROM_put)
 6. [EEPROM_read](examples/EEPROM_read)
 7. [EEPROM_update](examples/EEPROM_update)
 8. [EEPROM_write](examples/EEPROM_write)
 9. [EmulateEEPROM](examples/EmulateEEPROM)
10. [FlashStoreAndRetrieve](examples/FlashStoreAndRetrieve)
11. [StoreNameAndSurname](examples/StoreNameAndSurname)

### Examples from other libraries

### 1. Library [BlynkEthernet_WM](https://github.com/khoih-prog/BlynkEthernet_WM)

 4. [AM2315_W5500](https://github.com/khoih-prog/BlynkEthernet_WM/tree/master/examples/AM2315_W5500)
 5. [DHT11_W5500](https://github.com/khoih-prog/BlynkEthernet_WM/tree/master/examples/DHT11_W5500)
 6. [ENC28J60_Blynk](https://github.com/khoih-prog/BlynkEthernet_WM/tree/master/examples/ENC28J60_Blynk)
 7. [ENC28J60_Blynk_Email](https://github.com/khoih-prog/BlynkEthernet_WM/tree/master/examples/ENC28J60_Blynk_Email)
 8. [ENC28J60_WM_Config](https://github.com/khoih-prog/BlynkEthernet_WM/tree/master/examples/ENC28J60_WM_Config)
 9. [EthernetENC_Blynk_SAMD](https://github.com/khoih-prog/BlynkEthernet_WM/tree/master/examples/EthernetENC_Blynk_SAMD)
10. [W5500_Blynk](https://github.com/khoih-prog/BlynkEthernet_WM/tree/master/examples/W5500_Blynk)
11. [W5500_Blynk_Email](https://github.com/khoih-prog/BlynkEthernet_WM/tree/master/examples/W5500_Blynk_Email)
12. [W5500_WM_Config](https://github.com/khoih-prog/BlynkEthernet_WM/tree/master/examples/W5500_WM_Config)
13. [W5500_Blynk_Email_SAMD](https://github.com/khoih-prog/BlynkEthernet_WM/tree/master/examples/W5500_Blynk_Email_SAMD)
14. [W5500_Blynk_SAMD](https://github.com/khoih-prog/BlynkEthernet_WM/tree/master/examples/W5500_Blynk_SAMD)
15. [W5500_WM_Config_SAMD](https://github.com/khoih-prog/BlynkEthernet_WM/tree/master/examples/W5500_WM_Config_SAMD)
 
### 2. Library [Blynk_Esp8266AT_WM](https://github.com/khoih-prog/Blynk_Esp8266AT_WM)
 
16. [SAMD_ESP8266Shield](https://github.com/khoih-prog/Blynk_Esp8266AT_WM/tree/master/examples/SAMD_ESP8266Shield)
 
### 3. Library [Blynk_WiFiNINA_WM](https://github.com/khoih-prog/Blynk_WiFiNINA_WM)
 
17. [SAMD_WiFiNINA_WM](https://github.com/khoih-prog/Blynk_WiFiNINA_WM/tree/master/examples/SAMD_WiFiNINA_WM)
 
### 4. Library [ESP_AT_WiFiManager](https://github.com/khoih-prog/ESP_AT_WiFiManager) 
 
18. [AutoConnect](https://github.com/khoih-prog/ESP_AT_WiFiManager/tree/master/examples/AutoConnect)
19. [AutoConnectWithFeedback](https://github.com/khoih-prog/ESP_AT_WiFiManager/tree/master/examples/AutoConnectWithFeedback)
20. [ConfigOnStartup](https://github.com/khoih-prog/ESP_AT_WiFiManager/tree/master/examples/ConfigOnStartup)
21. [ConfigOnSwitch](https://github.com/khoih-prog/ESP_AT_WiFiManager/tree/master/examples/ConfigOnSwitch)

### 5. Library [ESP_AT_WM_Lite](https://github.com/khoih-prog/ESP_AT_WM_Lite) 
 
22. [SAMD_ESP8266Shield](https://github.com/khoih-prog/ESP_AT_WM_Lite/tree/master/examples/SAMD_ESP8266Shield)


### 6. Library [WiFiManager_NINA_Lite](https://github.com/khoih-prog/WiFiManager_NINA_Lite) 
 
23. [SAMD_WiFiNINA](https://github.com/khoih-prog/WiFiManager_NINA_Lite/tree/master/examples/SAMD_WiFiNINA)
24. [SAMD_WiFiNINA_MQTT](https://github.com/khoih-prog/ESP_AT_WM_Lite/tree/master/examples/SAMD_WiFiNINA_MQTT)

### 7. Library [DoubleResetDetector_Generic](https://github.com/khoih-prog/DoubleResetDetector_Generic) 
 
25. [minimal](https://github.com/khoih-prog/DoubleResetDetector_Generic/tree/master/examples/minimal)

### 8. Library [MultiResetDetector_Generic](https://github.com/khoih-prog/MultiResetDetector_Generic) 
 
26. [minimal](https://github.com/khoih-prog/MultiResetDetector_Generic/tree/main/examples/minimal)


---
---

### Example [StoreNameAndSurname](examples/StoreNameAndSurname)


```cpp
// Demonstrate how to use FlashStorage_SAMD with an API that is similar to the EEPROM library to Store and retrieve structured data.

#include <FlashAsEEPROM_SAMD.h>

const int WRITTEN_SIGNATURE = 0xBEEFDEED;

// Create a structure that is big enough to contain a name
// and a surname. The "valid" variable is set to "true" once
// the structure is filled with actual data for the first time.
typedef struct
{
  char name[100];
  char surname[100];
} Person;

void setup()
{
  Serial.begin(115200);
  while (!Serial);

  delay(200);

  Serial.print(F("\nStart StoreNameAndSurname on ")); Serial.println(BOARD_NAME);
  Serial.println(FLASH_STORAGE_SAMD_VERSION);

  Serial.print("EEPROM length: ");
  Serial.println(EEPROM.length());

  // Check signature at address 0
  int signature;

  // Create a "Person" variable and call it "owner"
  uint16_t storedAddress = 0;
  Person owner;

  EEPROM.get(storedAddress, signature);

  // If the EEPROM is empty then no WRITTEN_SIGNATURE
  if (signature == WRITTEN_SIGNATURE)
  {
    EEPROM.get(storedAddress + sizeof(signature), owner);

    // Say hello to the returning user!
    Serial.print("Hi "); Serial.print(owner.name); Serial.print(" "); Serial.print(owner.surname);
    Serial.println(", nice to see you again :-)");

    Serial.println("Clearing WRITTEN_SIGNATURE for next try");

    EEPROM.put(0, 0);
    Serial.println("Done clearing signature in emulated EEPROM. You can reset now");
  }
  else
  {
    Serial.println("EEPROM is empty, writing WRITTEN_SIGNATURE and some example data:");

    EEPROM.put(storedAddress, WRITTEN_SIGNATURE);

    // ...in this case we ask for user data.
    Serial.setTimeout(30000);
    Serial.print("Insert your name : ");
    String name = Serial.readStringUntil('\n');
    Serial.println(name);
    Serial.print("Insert your surname : ");
    String surname = Serial.readStringUntil('\n');
    Serial.println(surname);

    // Fill the "owner" structure with the data entered by the user...
    name.toCharArray(owner.name, 100);
    surname.toCharArray(owner.surname, 100);

    // ...and finally save everything into emulated-EEPROM
    EEPROM.put(storedAddress + sizeof(signature), owner);

    if (!EEPROM.getCommitASAP())
    {
      Serial.println("CommitASAP not set. Need commit()");
      EEPROM.commit();
    }

    // Print a confirmation of the data inserted.
    Serial.print("<< Your name: "); Serial.print(owner.name);
    Serial.print(". Your surname: "); Serial.print(owner.surname);
    Serial.println(" >> have been saved. Thank you!");
    Serial.println("You can reset to check emulated-EEPROM data retention.");
  }
}

void loop()
{
  // Do nothing...
}
```

---
---

### Debug Terminal Output Samples

### 1. W5500_Blynk on Adafruit SAMD51 ITSYBITSY_M4 using W5500 Ethernet shield


The following is the sample terminal output when running example [**W5500_Blynk**](https://github.com/khoih-prog/BlynkEthernet_WM/tree/master/examples/W5500_Blynk) on **Adafruit SAMD51 ITSYBITSY_M4 using W5500 Ethernet shield**

#### 1.1. No Config Data with LOAD_DEFAULT_CONFIG_DATA = false => Config Portal don't loads default Credential

```
Start W5500_Blynk on ITSYBITSY_M4
[936] ChkCrR:CrCCSum=0xaf50,CrRCSum=0xffffffff
[936] CCSum=0x0,RCSum=0x0
[936] Invalid Stored Dynamic Data. Load default from Sketch
[937] SaveEEPROM,Sz=1024,DataSz=0,WCSum=0x1d4d
[944] CrCCSum=0x29a6
[944] MAC:FE-A8-80-C6-CE-A3
_pinCS = 0
W5100 init, using SS_PIN_DEFAULT = 10, new ss_pin = 10, W5100Class::ss_pin = 1
W5100::init: W5500, SSIZE =8192
[2632] IP:192.168.2.153
[2633] b:Stay in CfgPortal:No CfgDat
[2633] CfgIP=192.168.2.153
F
Your stored Credentials :
MQTT Server = default-mqtt-server
Port = 1883
MQTT UserName = default-mqtt-username
MQTT PWD = default-mqtt-password
Subs Topics = default-mqtt-SubTopic
Pubs Topics = default-mqtt-PubTopic
FFFFF
[339285] SaveEEPROM,Sz=1024,DataSz=0,WCSum=0x2e89
[339292] CrCCSum=0x219f
```

#### 1.2. Input valid credentials with LOAD_DEFAULT_CONFIG_DATA = false. Click `Save` => reboot

```
Start W5500_Blynk on ITSYBITSY_M4
[1547] ChkCrR:CrCCSum=0x219f,CrRCSum=0x219f
[1547] CCSum=0x0,RCSum=0x0
[1548] CrCCSum=0x219f,CrRCSum=0x219f
[1548] ======= Start Stored Config Data =======
[1548] Hdr=W5X00,BName=Seeeduino_W5500_BlynkWM
[1548] Svr=account.duckdns.org,Tok=new_token1
[1549] Svr1=account.ddns.net,Tok1=new_token2
[1549] Prt=8080,SIP=192.168.2.220
[1549] connectEthernet: Use static_IP=192.168.2.220
[1549] MAC:FE-A1-D4-BC-E8-CB

W5100 init, using SS_PIN_DEFAULT = 10, new ss_pin = 10, W5100Class::ss_pin = 1
W5100::init: W5500, SSIZE =8192
[3131] IP:192.168.2.220
[3131] bg:ECon.TryB
[3131] 
    ___  __          __
   / _ )/ /_ _____  / /__
  / _  / / // / _ \/  '_/
 /____/_/\_, /_//_/_/\_\
        /___/ v0.6.1 on Arduino Zero

[3132] BlynkArduinoClient.connect: Connecting to account.duckdns.org:8080
[3244] Ready (ping: 6ms).
[3311] Connected to BlynkServer=account.duckdns.org,Token=new_token1
[3311] bg:EBCon
Conn2Blynk: server = account.duckdns.org, port = 8080
Token = new_token1, IP = 192.168.2.220
B
Your stored Credentials :
MQTT Server = new-mqtt-server
Port = 1883
MQTT UserName = new-mqtt-username
MQTT PWD = new-mqtt-password
Subs Topics = new-mqtt-SubTopic
Pubs Topics = new-mqtt-PubTopic
BBBBBBBBB BBBBBBBBBB BBBBBBBBBB BBBBBBBBBB BBBBBBBBBB BBBBBBBBBB BBBBBBBBBB
```

---

### 2. StoreNameAndSurname on SAMD21 SEEED_XIAO_M0


#### 2.1. No data => input then save

```
Start StoreNameAndSurname on SEEED_XIAO_M0
FlashStorage_SAMD v1.1.0
EEPROM length: 1024
EEPROM is empty, writing WRITTEN_SIGNATURE and some example data:
Insert your name : John

Insert your surname : Doe

<< Your name: John. Your surname: Doe >> have been saved. Thank you!
You can reset to check emulated-EEPROM data retention.
```

#### 2.2. Data retention after reset

```
Start StoreNameAndSurname on SEEED_XIAO_M0
FlashStorage_SAMD v1.1.0
EEPROM length: 1024
Hi John Doe, nice to see you again :-)
Clearing WRITTEN_SIGNATURE for next try
Done clearing signature in emulated EEPROM. You can reset now
```

---

### 3. EEPROM_Clear on SAMD21 SEEED_XIAO_M0

#### 3.1. 1ms to clear 1K bytes emulated-EEPROM

```
Start EEPROM_Clear on SEEED_XIAO_M0
FlashStorage_SAMD v1.1.0

Emulated EEPROM length (bytes) = 1024

Done clearing emulated EEPROM. Time spent (ms) = 11

```

#### 3.2. 22ms to clear 2K bytes emulated-EEPROM

```
Start EEPROM_Clear on SEEED_XIAO_M0
FlashStorage_SAMD v1.1.0
Emulated EEPROM length (bytes) = 2048
Done clearing emulated EEPROM. Time spent (ms) = 22
```

#### 3.3. 42ms to clear 4K bytes emulated-EEPROM

```
Start EEPROM_Clear on SEEED_XIAO_M0
FlashStorage_SAMD v1.1.0
Emulated EEPROM length (bytes) = 4096
Done clearing emulated EEPROM. Time spent (ms) = 42
```

---

### 4. EEPROM_get on SAMD21 SEEED_XIAO_M0

#### 4.1. No valid data, write signature and data

```
Start EEPROM_get on SEEED_XIAO_M0
FlashStorage_SAMD v1.1.0
EEPROM length: 1024
EEPROM doesn't store valid data, writing WRITTEN_SIGNATURE and some example data
Float written to EEPROM: 123.456
Done writing custom object to EEPROM: 
===============
Field1: 3.14159
Field2: 65
Name: Working!
===============
Reset to see how you can retrieve the values by using EEPROM_get!
```

#### 4.2. Valid signature and data, read data

```
Start EEPROM_get on SEEED_XIAO_M0
FlashStorage_SAMD v1.1.0
EEPROM length: 1024
EEPROM has valid data with WRITTEN_SIGNATURE. Now read some example data
Read float from EEPROM: 123.456
Read custom object from EEPROM: 
===============
Field1: 3.14159
Field2: 65
Name: Working!
===============
```

---
---


## FAQ

### Can I use a single object to store more stuff?

Yes, you can declare a `struct` with more fields and call a `EEPROM.put()` to store the entire structure. See the [StoreNameAndSurname](examples/StoreNameAndSurname) for how to do it.

### The content of the FlashStorage is erased each time a new sketch is uploaded?

Yes, every time you upload a new sketch, the previous content of the FlashStorage is erased.

### Do you recommend to use FLASH instead of EEPROM?

No. If your board provides an integrated-EEPROM, it's advisable to use that because EEPROM has longer lifetime, number of write cycles, etc.).

In the absence of an integrated-EEPROM or its size is too small for your use-case, you can use this library to use a small portion flash memory as emulated-EEPROM, provided that you keep in mind the limits as in [Limited number of writes](#limited-number-of-writes)

---
---

### Troubleshooting

If you get compilation errors, more often than not, you may need to install a newer version of the core for Arduino boards.

Sometimes, the library will only work if you update the board core to the latest version because I am using newly added functions.

---
---

## Releases

### Releases v1.1.0

1. Add supports to `EEPROM.put()` and `EEPROM.get()` for efficiently writing and reading the whole object. 
2. Add `_commitASAP` flag to permit reduce the number of calls to `EEPROM.commit()` to extend the flash life.
3. Add many modified examples from [Arduino EEPROM library](https://www.arduino.cc/en/Reference/EEPROM)
4. Fix bug.

### Releases v1.0.0

1. Add support to SAMD51 family such as 

  - Adafruit SAMD51 (M4): Metro M4, Grand Central M4, ItsyBitsy M4, Feather M4 Express, Trellis M4, Metro M4 AirLift Lite, MONSTER M4SK Express, Hallowing M4, etc.
  - Seeeduino: Wio Terminal, Grove UI WirelessAdafruit Itsy-Bitsy M4, etc.

---

### Supported hardware

### Currently supported Boards

  - **Arduino SAMD21 (ZERO, MKR, NANO_33_IOT, etc.)**.
  - **Adafruit SAM21 (Itsy-Bitsy M0, Metro M0, Feather M0, Gemma M0, etc.)**.
  - **Adafruit SAM51 (Itsy-Bitsy M4, Metro M4, Grand Central M4, Feather M4 Express, etc.)**.
  - **Seeeduino SAMD21/SAMD51 boards (SEEED_WIO_TERMINAL, SEEED_FEMTO_M0, SEEED_XIAO_M0, Wio_Lite_MG126, WIO_GPS_BOARD, SEEEDUINO_ZERO, SEEEDUINO_LORAWAN, SEEED_GROVE_UI_WIRELESS, etc.)** 

---
---

### Issues

Submit issues to: [FlashStorage_SAMD issues](https://github.com/khoih-prog/FlashStorage_SAMD/issues)

---

### TO DO

1. Search for bug and improvement.
2. Similar features for remaining Arduino boards

---

### DONE

1. Basic emulated-EEPROM for SAMD21/SAMD51.
2. Add `EEPROM.put()` and `EEPROM.get()` functions to read/write the whole struct in emulated-EEPROM
3. Similar features for STM32F/L/H/G/WB/MP1.
4. Add Table of Contents

---
---

### Contributions and thanks

1. Based on and modified from [Cristian Maglie's FlashStorage](https://github.com/cmaglie/FlashStorage). Most of the credits go to [Cristian Maglie](https://github.com/cmaglie)

<table>
  <tr>
    <td align="center"><a href="https://github.com/cmaglie"><img src="https://github.com/cmaglie.png" width="100px;" alt="cmaglie"/><br /><sub><b>⭐️⭐️ Cristian Maglie</b></sub></a><br /></td>
  </tr> 
</table>

---

## Contributing

If you want to contribute to this project:
- Report bugs and errors
- Ask for enhancements
- Create issues and pull requests
- Tell other people about this library

---


### License

- The library is licensed under [LGPLv3](https://github.com/khoih-prog/FlashStorage_SAMD/blob/master/LICENSE)



