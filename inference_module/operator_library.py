'''
Title: DSL Library of Dalek Mind
Author: Belegurth Meleko
Date: September 16 2021
'''
import tensorflow as tf # Install Common Utils of Tensorflow
import tensorflow.keras as keras # Use the Keras API to Create Models
import tensorflow.keras.layers as layers # Import Dense Layers

OperatorList = ["Advance","Turn","TurnAndAdvance","Fire","STOP"]
Operator_Id = tf.convert_to_tensor([0,1,2,3,4])
Arguments_Id = tf.convert_to_tensor([0,1,2,3,4])
ArgumentDiction = {"Advance":[0],
                    "Turn":[1],
                    "TurnAndAdvance":[2,3],
                    "Fire":[4]}
# Set up the basic
OperatorWithArguments = ["Advance","Turn","TurnAndAdvance","Fire"]

num_op = len(OperatorList)
num_arg = len(Arguments_Id)
# Create an operators embedder : Turn each operator ——> R(E)
OperatorEmbeder = keras.Sequential()
OperatorEmbeder.add(layers.Embedding(num_op,64,input_length = 1))

#Create an arguments embedder : Turn each argument ——> R(E)
ArgEmbeder = keras.Sequential()
ArgEmbeder.add(layers.Embedding(num_arg,64,input_length = 1))

def name2index(name,type):
    if type == "operator":
        return [OperatorList.index(name)]
    if type == "arg":
        return ArgumentDiction[name]

def index2embed(indices,type):
    indices = tf.convert_to_tensor(indices)
    if type == "arg":
        return ArgEmbeder(indices)
    if type == "operator":
        return OperatorEmbeder(indices)


def DoAnd(program1,program2):
    exec(program1)
    exec(program2)

def Advance(distance):
    print("Dalek Advanced for {}".format([distance]))

def Turn(angle):
    print("Turned angle of {}".format([angle]))

def TurnAndAdvance(angle,distance):
    Turn(angle)
    Advance(distance)

def STOP():
    return "STOP"

def Left():
    return "LEFT"

def Right():
    return "RIGHT"

def Back():
    return "BACK"

def Face():
    return "FACE"

def Fire():
    print("Exterminate!")