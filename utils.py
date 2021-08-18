# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 18:51:45 2021

@author: Acer
"""

import numpy as np
# Symmetric Log:
def Sym_Log(x):
    if x>0:
        return np.log(x)
    elif x < 0:
        return -np.log(-x)
    else:
        return 0

# Positive log:
def Pos_Log(x):
    if x>0:
        return np.log(x)
    else:
        return 0