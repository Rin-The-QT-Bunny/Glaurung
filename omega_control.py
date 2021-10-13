"""
Omega Controller of Dalek Mind
"""

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# The idea behind it is to create more 

class Omega:
    # a Omega class that is used to control all alphas
    def __init__(self):
        print("Omega Control is created")
        self.name = "Mec'thuen" # That's why we named it Mec'thuen, dude.
        self.alphas = list() # an omega has a set of alphas.
        self.GEncoder = None; # Use the GEncoder to do the selection
    
    def runAlpha(self,alpha_id,tasks):
        # use an omega to solve a task
        pass

    def createAlpha(self):
        # when current alphas are not capable of handling tasks given.
        # Omega will just create one more alpha and let is solve the remaining task
        print("An Alpha is just created")