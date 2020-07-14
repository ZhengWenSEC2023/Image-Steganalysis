# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 10:44:47 2020

@author: Lenovo
"""

import pickle

f = open('G:\image_stego_code\p2_original.pkl', 'rb')
p2_original = pickle.load(f)
f.close()

f = open('G:\image_stego_code\kernels_original.pkl', 'rb')
kernels_original = pickle.load(f)
f.close()

f = open('G:\image_stego_code\p2_stego.pkl', 'rb')
p2_stego = pickle.load(f)
f.close()

f = open('G:\image_stego_code\kernels_stego.pkl', 'rb')
kernels_stego = pickle.load(f)
f.close()