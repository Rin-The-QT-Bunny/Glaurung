'''
Title: Neural Models of Dalek Mind
Author: Belegurth Meleko
Date: September 16 2021
'''

import tensorflow as tf # Install Common Utils of Tensorflow
import tensorflow.keras as keras # Use the Keras API to Create Models
import tensorflow.keras.layers as layers # Import Dense Layers

# Set Up dimensions and other factors

operator_dim = 64 # dimension of opertor embeding
semantics_dim  = 64 # dimension of semantics embedding 
arg_dim = 64 # dimension of argument embedding
max_words = 2000 # Maximum words in the vocabulary
dim_words = 32
max_len = 30 # Maximum words input into the sentence

# Create the Probability Evaluator

op_input = keras.Input(shape = (operator_dim))
s_input = keras.Input(shape = (semantics_dim))
conc_ops = layers.concatenate([op_input,s_input])

#  Dense (Full Connect) Layer that imitates the prob
hidden_p1 = layers.Dense(32,"tanh")(conc_ops)
hidden_p2 = layers.Dense(36,"tanh")(hidden_p1)
hidden_p3 = layers.Dense(36,"tanh")(hidden_p2)
prob_output = layers.Dense(1,"softplus")(hidden_p3) # Use the sigmoid to output prob

# Connect inputs and outputs to create the proabbility evaluator
P = keras.Model([op_input,s_input],prob_output)

# Create the Semantics Repeater

order_input = keras.Input(shape = (semantics_dim))
arg_input = keras.Input(shape = (arg_dim))
conc_oa = layers.concatenate([order_input,arg_input])

# Semantics Repeater that pass down the semantics information
hidden_r1 = layers.Dense(64,"tanh")(conc_oa)
hidden_r2 = layers.Dense(36,"tanh")(hidden_r1)
hidden_r3 = layers.Dense(36,"tanh")(hidden_r2)
semantics_output = layers.Dense(2 * semantics_dim,"tanh")(hidden_r3) # Use the tanh to output semantics signal 

# Connect inputs and outputs to create the semantics repeater
R = keras.Model([order_input,arg_input],semantics_output)

# Sentence Embedding Model

sequence_input = keras.Input(shape = [max_len,dim_words])
gate_unit = layers.GRU(semantics_dim)(sequence_input)
GEncoder = keras.Model(sequence_input,gate_unit)


# The word embeder created
WordEmbeder = keras.Sequential()
WordEmbeder.add(layers.Embedding(max_words,dim_words))

P.summary()
R.summary()
GEncoder.summary()
WordEmbeder.summary()