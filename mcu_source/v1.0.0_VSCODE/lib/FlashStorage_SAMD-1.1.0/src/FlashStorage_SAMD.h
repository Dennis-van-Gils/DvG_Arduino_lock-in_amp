/******************************************************************************************************************************************
  FlashStorage_SAMD.h
  For SAMD21/SAMD51 using Flash emulated-EEPROM

  The FlashStorage_SAMD library aims to provide a convenient way to store and retrieve user's data using the non-volatile flash memory
  of SAMD21/SAMD51. It now supports writing and reading the whole object, not just byte-and-byte.

  Based on and modified from Cristian Maglie's FlashStorage (https://github.com/cmaglie/FlashStorage)

  Built by Khoi Hoang https://github.com/khoih-prog/FlashStorage_SAMD
  Licensed under LGPLv3 license
  
  Orginally written by Cristian Maglie
  
  Copyright (c) 2015 Arduino LLC.  All right reserved.
  Copyright (c) 2020 Khoi Hoang.
  
  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
  as published bythe Free Software Foundation, either version 3 of the License, or (at your option) any later version.
  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
  You should have received a copy of the GNU Lesser General Public License along with this library. 
  If not, see (https://www.gnu.org/licenses/)
  
  Version: 1.1.0

  Version Modified By   Date        Comments
  ------- -----------  ----------   -----------
  1.0.0   K Hoang      28/03/2020  Initial coding to add support to SAMD51 besides SAMD21
  1.1.0   K Hoang      26/01/2021  Add supports to put() and get() for writing and reading the whole object. Fix bug.
 ******************************************************************************************************************************************/

#pragma once

#ifndef FlashStorage_SAMD_h
#define FlashStorage_SAMD_h

#define FLASH_STORAGE_SAMD_VERSION     "FlashStorage_SAMD v1.1.0"

#include <Arduino.h>

// Concatenate after macro expansion
#define PPCAT_NX(A, B) A ## B
#define PPCAT(A, B) PPCAT_NX(A, B)

#if defined(__SAMD51__)
  #define Flash(name, size) \
  __attribute__((__aligned__(8192))) \
  static const uint8_t PPCAT(_data,name)[(size+8191)/8192*8192] = { }; \
  FlashClass name(PPCAT(_data,name), size);

#define FlashStorage(name, T) \
  __attribute__((__aligned__(8192))) \
  static const uint8_t PPCAT(_data,name)[(sizeof(T)+8191)/8192*8192] = { }; \
  FlashStorageClass<T> name(PPCAT(_data,name));
  
#else

#define Flash(name, size) \
  __attribute__((__aligned__(256))) \
  static const uint8_t PPCAT(_data,name)[(size+255)/256*256] = { }; \
  FlashClass name(PPCAT(_data,name), size);

#define FlashStorage(name, T) \
  __attribute__((__aligned__(256))) \
  static const uint8_t PPCAT(_data,name)[(sizeof(T)+255)/256*256] = { }; \
  FlashStorageClass<T> name(PPCAT(_data,name));

#endif

class FlashClass {
public:
  FlashClass(const void *flash_addr = NULL, uint32_t size = 0);

  void write(const void *data) { write(flash_address, data, flash_size); }
  void erase()                 { erase(flash_address, flash_size);       }
  void read(void *data)        { read(flash_address, data, flash_size);  }

  void write(const volatile void *flash_ptr, const void *data, uint32_t size);
  void erase(const volatile void *flash_ptr, uint32_t size);
  void read(const volatile void *flash_ptr, void *data, uint32_t size);

private:
  void erase(const volatile void *flash_ptr);

  const uint32_t PAGE_SIZE, PAGES, MAX_FLASH, ROW_SIZE;
  const volatile void *flash_address;
  const uint32_t flash_size;
};

template<class T>
class FlashStorageClass {
public:
  FlashStorageClass(const void *flash_addr) : flash(flash_addr, sizeof(T)) { };

  // Write data into flash memory.
  // Compiler is able to optimize parameter copy.
  inline void write(T data)  { flash.erase(); flash.write(&data); }

  // Read data from flash into variable.
  inline void read(T *data)  { flash.read(data); }

  // Overloaded version of read.
  // Compiler is able to optimize copy-on-return.
  inline T read() { T data; read(&data); return data; }

private:
  FlashClass flash;
};

#if    ( defined(__SAMD51__) || defined(__SAMD51J20A__) || defined(__SAMD51J19A__) || defined(__SAMD51G19A__)  )
#include "FlashStorage_SAMD51.h"
#elif    ( defined(ARDUINO_SAMD_ZERO) || defined(ARDUINO_SAMD_MKR1000) || defined(ARDUINO_SAMD_MKRWIFI1010) \
      || defined(ARDUINO_SAMD_NANO_33_IOT) || defined(ARDUINO_SAMD_MKRFox1200) || defined(ARDUINO_SAMD_MKRWAN1300) || defined(ARDUINO_SAMD_MKRWAN1310) \
      || defined(ARDUINO_SAMD_MKRGSM1400) || defined(ARDUINO_SAMD_MKRNB1500) || defined(ARDUINO_SAMD_MKRVIDOR4000) || defined(__SAMD21G18A__) \
      || defined(ARDUINO_SAMD_CIRCUITPLAYGROUND_EXPRESS) || defined(__SAMD21E18A__) )
#include "FlashStorage_SAMD21.h"
#else
#error This code is intended to run only on the SAMD21 or SAMD51 boards ! Please check your Tools->Board setting.
#endif    
      
#endif    //#ifndef FlashStorage_SAMD_h
