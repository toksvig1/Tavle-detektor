# De nødvendige bibloteker bliver importeret her.
import numpy as np # Til matematik
import sys # Systemet
import matplotlib # 

# Det neurale netværk

class tmp:
    def __init__(self, inputs, weights, bias):
        self.i = inputs
        self.w = weights
        self.b = bias


    def NCalc(self):
        output = self.i[0]*self.w[0]+self.i[1]*self.w[1]+self.i[2]*self.w[2]+ self.b
        self.o = output




def main():
    # Versioner printes
    print("Python version:", sys.version)
    print("numpy version:", np.__version__)
    print("Matplotlib version:", matplotlib.__version__)
    inputs = [1.2,5.1,2.1]
    weights = [3.1,2.1,8.7]
    bias = 3
    neuron = tmp(inputs,weights,bias)
    neuron.NCalc()
    print(neuron.o)




# Programmet kan kun køres som script, og kan ikke importeres i andre scripts.
if __name__ == "__main__":
    main()