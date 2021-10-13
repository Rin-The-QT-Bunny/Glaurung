'''
Title: Tokenizer tbe of Dalek Mind
Author: Belegurth Meleko
Date: September 16 2021
'''

import tensorflow as tf # Install Common Utils of Tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer

dim_words = 32
# Create the Tokenizer with the maximum words

max_words = 2000 # Maximum words in the vocabulary
max_len = 30 # Maximum words input into the sentence

# Load the text file with multiple lines
def Load_txt(File):
    return_file = []
    fopen=open(File,'r')
    lines=fopen.readlines()
 
    for line in lines:
        line = line.strip('\n')
        # Take out the line signal here

        return_file.append(line)
    return return_file

# Regularize the input of the query
def Fixlength(seqs):
    for i in range(len(seqs)):
        
        # process for each sequence in the input commands
        process_seq = seqs[i]

        # While the input sequence is longer than regular sequence
        if len(process_seq) > max_len:
            # Cut the input sequence 
            process_seq = process_seq[:max_len]
        
        else:
            # the input sequence is shorter than regular sequence
            while len(process_seq) < max_len:
                # Use a zero element to padd the sequence
                process_seq.append(0)
        
        seqs[i] = process_seq
    return seqs

def FilterNum(string):
    listed_elements = list(string)
    for i in range(len(listed_elements)):
        try:
            int(listed_elements[i])
            listed_elements[i] = "num"
        except:
            pass
    return_string = ""
    for item in listed_elements:
        return_string+=item
    return return_string

# Regularizer:

def renormalize(sequence,max_len):
    return_sequence = []
    for i in range(max_len):
        try:
            return_sequence.append(sequence[0][i])
        except:
            return_sequence.append(0)
    return return_sequence
#Tokenizer Instance
tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')


# testing raw vocabulary material
material = ["A red ball is at the top of the building",
 "The building is next to the man",
 "The sun is burning bright",
 "The eye of Morgoth is red as the sun",
 "I saw a man shoot at a bird",
 "Advance for 5 meters and stop",
 "Turn right and stop.",
 "Turn and advance forward",
 "Go right and stop","Turn(STOP())",
 "Go right and wait","Turn(STOP())",
 "Turn around",
 "Walk forward for 5 meters",
 "Walk forward",
 "Go backward",
 "Turn right and advance forward",
 "Advance and stop",
 "Go back",
 "Turn around about",
 "Advance for 5 meters",
"Turn to the right and advance forward",
"Turn right and advance",
"Turn right and go forward",
"Turn right and advance forward",
"Go forward.",
"Advance forward",
"Advance forward for 5 meters",
"Turn around",
"Turn to your face",
"Turn around and face forward.",
"Face forward",
"Turn(Face())"
 ]

tokenizer.fit_on_texts(material)

def GRU_Module(query,GRU,wordEmbeder):
    sequence = tokenizer.texts_to_sequences([query])
    sequence = renormalize(sequence,max_len)

    sequence = tf.reshape(sequence,[1,max_len,-1])
    embed = wordEmbeder(sequence)
    embed = tf.reshape(embed,[1,max_len,dim_words])

    return GRU(embed)
