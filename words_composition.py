'''
Title: Master Processer of Dalek Mind
Author: Belegurth Meleko
Date: September 16 2021
'''

from inference_module.neural_models import *
from inference_module.sentence_tokenizer import *

# Load the data of all commands input
commands = Load_txt("data/sentences/commands.txt")

# Tokenize the texts and tansfer into onhot vector
tok.fit_on_texts(commands)
seqs = tok.texts_to_sequences(commands)

# The regular length of a sequence is 30 and shorter sequencd pad with 10
seqs = Fixlength(seqs)
seqs = tf.convert_to_tensor(seqs) # Tensorflow only allow tensor input

# Embeder called to transfer one-hot vectors to vector form
embeds = WordEmbeder(seqs)

semantics_dim = GEncoder(embeds)