'''
Title: Program Synthesis of Dalek Mind
Author: Celerinsil Meleko
Date: September 19 2021
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from inference_module.neural_models import *
from inference_module.parser_trial import *
from inference_module.operator_library import *
from inference_module.sentence_tokenizer import *

def filter(List):
    raw = []
    for i in range(len(List)):
        if List[i] == " " or List[i] == "":
            pass
        else:
            raw.append(List[i])
    return raw

def operator_bank():
    raw = []
    for i in range(len(OperatorList)):
        op_idx = name2index(OperatorList[i],"operator")
        raw.append([OperatorList[i],index2embed(op_idx,"operator")])
    return raw

def translate_pdf(raw):
    return_raw = []
    for i in range(len(raw)):
        return_raw.append(raw[i].numpy())
    return return_raw

class ProgramSynthesis:
    def __init__(self,name,op_selector,repeater):
        self.name = name
        print("The Representation Manifested is "+ str(self.name))
        self.operator_selector = op_selector
        self.semantics_repeater = repeater
        self.GAT = None
        
        # Hyper Parameters
        self.program = ""
        self.level = 0
        self.max_level = 1
        self.count = 0
        
        # Training Objective
        self.synthesis_log_probability = 0

        # the memory bank of operators
        self.operator_dictionary = None
        """
        Format(operator_dictionary){
            [op1, vector1],
            [op2, vector2],
            [op3, vector3],
            ... ...
            [op_n, vector_n]
        }
        """

        print("Analyzer Module Environment, Online.")
        
    def GainOperator(self,semantics_rep,DSL):

        self.program = "" # intiate the set up of programs: empty program. 
        self.synthesis_log_probability = 0 # optimization parameter: logP(program)

        operator_activation = [] # use this to record the probability distribution of each operators.
        self.operator_dictionary = DSL
        
        def convert(semantics,level):
            if level <= 5:
                l= level + 1
                size = len(self.operator_dictionary)

                semantics = tf.reshape(semantics,[1,semantics_dim])
                operator_activation = []
                for i in range(len(self.operator_dictionary)):
                    op = tf.reshape(self.operator_dictionary[i][1],[1,semantics_dim])
                    semantics = tf.reshape(semantics,[1,semantics_dim])
                    operator_activation.append(self.operator_selector([op,semantics])[0][0])


                pdf = translate_pdf(operator_activation[-size:])
                #Select the operator according to PDF
                #Index = np.argmax(pdf)
                Index = np.random.choice(range(len(self.operator_dictionary)),p = pdf/sum(pdf))
                operator = self.operator_dictionary[Index][0]

                self.program += operator
                    
                # Calculate the Probability
                
                # Re Do the Probability Term w.r.t Op
 
                self.synthesis_log_probability += -tf.math.log(operator_activation[Index]/tf.reduce_sum(operator_activation))
                self.program += "("
                # Continue the Operator selection
                if operator in OperatorWithArguments : #If the operator requires arguments, then continue.
                    
                    args = name2index(operator,"arg")
                    # Creat the Bifurcation for each Arguemnt
                    for i in range(len(args)):
                        argument = args[i]
                        #Parse Each Argument
                        arg = index2embed(argument,"arg")
                        arg = tf.reshape(arg,[1,-1])

                        parameters = self.semantics_repeater([arg,semantics])
                        # Generate the Next Semantics Representation
                        target = parameters[0][:semantics_dim]
                        sigma = parameters[0][semantics_dim:2* semantics_dim]
                        
                        nxt_semantics = tf.random.normal([1],target,sigma)
                        self.synthesis_log_probability -= tf.math.log(tf.reduce_mean(tf.exp(-(nxt_semantics-target)**2)))

                        convert(nxt_semantics,l)

                        if i != (len(args)-1):
                            self.program+=","
                        # add a "," of each argument that is not at the end of it.
                self.program += ")" # add a ket at the end of the parsing

            else:self.program += "Rand"
        convert(semantics_rep,0)

    

        return self.program, self.synthesis_log_probability

    def TargetTree(self,semantics_rep,DSL,target):
        ops_sequence = Decompose(target)
        ops_sequence = filter(ops_sequence)

        self.program = "" # intiate the set up of programs: empty program. 
        self.synthesis_log_probability = 0 # optimization parameter: logP(program)
        #print(ops_sequence)
        operator_activation = [] # use this to record the probability distribution of each operators.
        self.operator_dictionary = DSL
        self.count = 0
        
        def convert(semantics,level):
            if level <= 5:
                l= level + 1
                size = len(self.operator_dictionary)

                semantics = tf.reshape(semantics,[1,semantics_dim])
                operator_activation = []
                for i in range(len(self.operator_dictionary)):
                    op = tf.reshape(self.operator_dictionary[i][1],[1,semantics_dim])
                    semantics = tf.reshape(semantics,[1,semantics_dim])
                    operator_activation.append(self.operator_selector([op,semantics])[0][0])


                pdf = translate_pdf(operator_activation[-size:])
                #Select the operator according to PDF
                Index = OperatorList.index(ops_sequence[self.count])

                #print(operator_activation)
                #print(Index,self.operator_dictionary[Index][0])
                #print(translate_pdf(operator_activation))
                self.count += 1
                operator = self.operator_dictionary[Index][0]

                self.program += operator
                    
                # Calculate the Probability
                
                # Re Do the Probability Term w.r.t Op
                #print(pdf)
 
                self.synthesis_log_probability += -tf.math.log(operator_activation[Index]/tf.reduce_sum(operator_activation))

                # Continue the Operator selection
                self.program += "("

                if operator in OperatorWithArguments : #If the operator requires arguments, then continue.

                    args = name2index(operator,"arg")
                    # Creat the Bifurcation for each Arguemnt
                    for i in range(len(args)):
                        argument = args[i]
                        #Parse Each Argument
                        arg = index2embed(argument,"arg")
                        arg = tf.reshape(arg,[1,-1])

                        parameters = self.semantics_repeater([arg,semantics])
                        # Generate the Next Semantics Representation
                        target = parameters[0][:semantics_dim]
                        sigma = parameters[0][semantics_dim:2* semantics_dim]
                        
                        nxt_semantics = tf.random.normal([1],target,sigma)
                        self.synthesis_log_probability += -tf.math.log(tf.reduce_mean(tf.exp(-(nxt_semantics-target)**2)))

                        convert(nxt_semantics,l)

                        if i != (len(args)-1):
                            self.program+=","
                        # add a "," of each argument that is not at the end of it.
                self.program += ")" # add a ket at the end of the parsing

            else:self.program += "Rand"
        convert(semantics_rep,0)

    

        return self.program, self.synthesis_log_probability
    
    def save_model(self):
        P.save_weights("parameters/P.h5")
        R.save_weights("parameters/R.h5")
        OperatorEmbeder.save_weights("parameters/OperatorEmbeder.h5")
        ArgEmbeder.save_weights("parameters/ArgEmbeder.h5")
        GEncoder.save_weights("parameters/GEncoder.h5")
        WordEmbeder.save_weights("parameters/WordEmbeder.h5")
    
    def load_model(self):
        P.load_weights("parameters/P.h5")
        R.load_weights("parameters/R.h5")
        OperatorEmbeder.load_weights("parameters/OperatorEmbeder.h5")
        ArgEmbeder.load_weights("parameters/ArgEmbeder.h5")
        GEncoder.load_weights("parameters/GEncoder.h5")
        WordEmbeder.load_weights("parameters/WordEmbeder.h5")


def train_tasks(tasks,EPOCH,eta):
    Adam = tf.optimizers.Adam(eta)
    history = []
    r_history  = []
    for epoch in range(EPOCH):
        reward = 0
        # Synthesis the program and watch the probability w.r.t each components
        with tf.GradientTape(persistent = True) as tape:
            logP = 0
            for i in range(len(tasks)):
                semantics = GRU_Module(tasks[i][0],GEncoder,WordEmbeder)
                pro, log = syns.GainOperator(semantics,operator_bank())
                #program, lnP = syns.TargetTree(semantics,operator_bank(),tasks[i][1])
                #print("sample program",pro)
                #print(pro)
                #print("target program",program)
                if pro == tasks[i][1]:
                    r = 1
                else:
                    r = -0.1
                logP += log * r 
                reward += r/len(tasks)
            logP = logP / len(tasks)
        history.append(logP)
        r_history.append(reward)
        # Calculate the gradient of LogP w.r.t to each components
        gradP = tape.gradient(logP,P.variables)
        gradR = tape.gradient(logP,R.variables)
        grad_op = tape.gradient(logP,OperatorEmbeder.variables)
        grad_arg = tape.gradient(logP,ArgEmbeder.variables)
        grad_GRU = tape.gradient(logP,GEncoder.variables)
        grad_word = tape.gradient(logP,WordEmbeder.variables)
        # Use the optimizer to ascdent the graident components
        try:
            Adam.apply_gradients(zip(gradR,R.variables))
        except:
            print("One Order Logic, No Gradient Provided")
            pass
        try:
            Adam.apply_gradients(zip(grad_arg,ArgEmbeder.variables))
        except:
            print("One Order Logic, No Gradient Provided")
            pass

        Adam.apply_gradients(zip(gradP,P.variables))
        Adam.apply_gradients(zip(grad_op,OperatorEmbeder.variables))
        Adam.apply_gradients(zip(grad_GRU,GEncoder.variables))
        Adam.apply_gradients(zip(grad_word,WordEmbeder.variables))
        # show it is the end of the epoch
        plt.cla()
        plt.plot(r_history)
        plt.pause(0.1)
        print("|Epoch: {} | Accuracy: {}| Average-LogP: {}|".format(epoch,reward,logP))
        if epoch%3 == 0:
            syns.save_model()
    plt.plot(history)


syns = ProgramSynthesis("Azari",P,R)

Tasks = [
    ["Advance for 5 meters","Advance(STOP())"],
    ["Go forward for 5 meters","Advance(STOP())"],
    ["Move forward for 5 meters","Advance(STOP())"],
    ["Advance","Advance(STOP())"],
    ["Go forward","Advance(STOP())"],
    ["Move forward","Advance(STOP())"],
    ["Turn around","Turn(STOP())"],
    ["Turn around about","Turn(STOP())"]

]

try:
    syns.load_model()
except:
    print("Failed to load models")

train_tasks(Tasks,526,0.000003)
syns.save_model()
plt.show()

for i in range(len(Tasks)):
    print("Order Given:",Tasks[i][0])
    print("Target Action:",Tasks[i][1])
    semantics = GRU_Module(Tasks[i][0],GEncoder,WordEmbeder)
    pro, log = syns.GainOperator(semantics,operator_bank())
        
    print("Sythesis Action:",pro)
    print("\n")

def weave(Question,GEncoder,WordEmbeder):
    semantics = GRU_Module(Question,GEncoder,WordEmbeder)
    pro, log = syns.GainOperator(semantics,operator_bank())
    return pro

print(weave("Advance and stop",GEncoder,WordEmbeder))

state = 0
while state == 0:
    order = input("Input the question: ")
    if order == "karakhzam":
        state = 1
        break
    else:
        ans = weave(order,GEncoder,WordEmbeder)
        print(ans)
        eval(ans)