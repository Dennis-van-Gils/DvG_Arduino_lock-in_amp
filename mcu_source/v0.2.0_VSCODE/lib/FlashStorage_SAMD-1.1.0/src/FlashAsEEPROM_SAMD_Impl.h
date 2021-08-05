/******************************************************************************************************************************************
  FlashAsEEPROM_SAMD_Impl.h
  For SAMD21/SAMD51 using Flash emulated-EEPROM

  The FlashStorage_SAMD library aims to provide a convenient way to store and retrieve user's data using the non-volatile flash memory
  of SAMD21/SAMD51. It now supports writing and reading the whole object, not just byte-and-byte.

  Based on and modified from Cristian Maglie's FlashStorage (https://github.com/cmaglie/FlashStorage)

  Built by Khoi Hoang https://github.com/khoih-prog/FlashStorage_SAMD
  Licensed under LGPLv3 license
  
  Orginally written by A. Christian
  
  Copyright (c) 2015-2016 Arduino LLC.  All right reserved.
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

#ifndef FlashAsEEPROM_SAMD_Impl_h
#define FlashAsEEPROM_SAMD_Impl_h

FlashStorage(eeprom_storage, EEPROM_EMULATION);

EEPROMClass::EEPROMClass() : _initialized(false), _dirty(false), _commitASAP(true)  
{
  // Empty
}

uint8_t EEPROMClass::read(int address)
{
  if (!_initialized) 
    init();
    
  return _eeprom.data[address];
}

void EEPROMClass::update(int address, uint8_t value)
{
  if (!_initialized) 
    init();
    
  if (_eeprom.data[address] != value) 
  {
    _dirty = true;
    _eeprom.data[address] = value;
  }
}

void EEPROMClass::write(int address, uint8_t value)
{
  update(address, value);
}


void EEPROMClass::init()
{
  _eeprom = eeprom_storage.read();
  
  if (_eeprom.signature != SAMD_EEPROM_EMULATION_SIGNATURE)
  {
    memset(_eeprom.data, 0xFF, EEPROM_EMULATION_SIZE);
    _eeprom.signature = SAMD_EEPROM_EMULATION_SIGNATURE;
  }
  
  _eeprom.valid = true;
   
  _initialized = true;
}
 
/**
 * Read from eeprom cells to an object
* @param index
* @param value
*/
//Functionality to 'get' data to objects to from EEPROM.
template< typename T > T& EEPROMClass::get( int idx, T &t )
{       
  // Copy the data from the flash to the buffer if not yet
  if (!_initialized) 
    init();
    
  uint16_t offset = idx;
  uint8_t* _pointer = (uint8_t *) &t;
  
  for ( uint16_t count = sizeof(T) ; count ; --count, ++offset )
  {  
    *_pointer++ = _eeprom.data[offset];
  }
    
  return t;
}

/**
* Read from eeprom cells to an object
* @param index
* @param value
*/
//Functionality to 'get' data to objects to from EEPROM.
template< typename T > const T& EEPROMClass::put( int idx, const T &t )
{            
  // Copy the data from the flash to the buffer if not yet
  if (!_initialized) 
    init();
  
  uint16_t offset = idx;
     
  const uint8_t* _pointer = (const uint8_t *) &t;
  
  for ( uint16_t count = sizeof(T) ; count ; --count, ++offset )
  {              
    _eeprom.data[offset] = *_pointer++;
  }

  if (_commitASAP)
  {
    _dirty = false;
    _eeprom.valid = true;
    // Save the data from the buffer
    eeprom_storage.write(_eeprom);
  }
  else  
  {
    // Delay saving the data from the buffer to the flash. Just flag and wait for commit() later
    _dirty = true;    
  }
       
  return t;
}

bool EEPROMClass::isValid()
{
  if (!_initialized) 
    init();
    
  return _eeprom.valid;
}

void EEPROMClass::commit()
{
  if (!_initialized) 
    init();
    
  if (_dirty) 
  {
    _dirty = false;
    _eeprom.valid = true;
    // Save the data from the buffer
    eeprom_storage.write(_eeprom);
  }
}

EEPROMClass EEPROM;

#endif    //FlashAsEEPROM_SAMD_Impl_h
