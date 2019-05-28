# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:46:52 2019

@author: vangi
"""

import numpy as np
import matplotlib.pyplot as plt

"""
           V_supply
               *
        R1  /     \ R2
        A  *       * B
 R_var = R4 \     / R3 = R_therm
               *
              GND


Real situation:
R1 = 22.00 kOhm
R2 = 22.19 kOhm

R_therm:
@ 25.0 'C    25.0 kOhm
@ 22.5 'C    

"""

V_supply = 1.4

R1 = 22000 #22000
R2 = 22000 #22190
# R_therm
R3 = np.arange(14000, 30000, 1, dtype=np.float)
# R_var
R4 = 22000 #21500

#AB = V_supply * (R2 * R4 - R1 * R3) / ((R1 + R4) * (R2 + R3))
AB = V_supply * (R1 / (R1 + R4) - R3 / (R3 + R2))

plt.plot(R3, AB, '.')

R3 = 22000
D_R1 = 0. #np.arange(-1000, 1000, dtype=np.float)
D_R2 = 0. #np.arange(-1000, 1000, dtype=np.float)
D_R3 = np.arange(-8000, 8000, dtype=np.float)
D_R4 = 0. #np.arange(-1000, 1000, dtype=np.float)
D_AB = (V_supply * R1 * R4 / (R1 + R4)**2 *
        (D_R1 / R1 - D_R2 / R2 + D_R3 / R3 - D_R4 / R4))

#plt.plot(D_R3, D_AB, '.')
