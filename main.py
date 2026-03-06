# De nødvendige bibloteker bliver importeret her.
import numpy as np # Til matematik
import sys # Systemet
import matplotlib # 
import random

# Parametre ################################################

HIDDEN_LAYERS = 0
INPUT_NODES = 1
HIDDEN_LAYERNODES = 20
OUTPUT_NODES = 1




# Parametre ################################################


# Det neurale netværk

class network:
    def __init__(self,hidden_layersamt, input_nodes, hidden_layernodes, output_nodes):
        self.hidden_layersamt = hidden_layersamt
        self.input_nodes = input_nodes
        self.hidden_layernodes = hidden_layernodes
        self.output_nodes = output_nodes
        self.hidden_layers = []
        self.input_layer = []
        self.output_layer = []
        self.create_layers()
        

    def create_layers(self):
        self.input_layer.append(layer(self.input_nodes,"input_layer",0))
        for x in range(self.hidden_layersamt):
            if x == 0:
                self.hidden_layers.append(layer(self.hidden_layernodes, "hidden_layer"+str(x+1),self.input_nodes))
            else:
                self.hidden_layers.append(layer(self.hidden_layernodes, "hidden_layer"+str(x+1),self.hidden_layernodes))
        self.output_layer.append(layer(self.output_nodes, "output_layer",self.hidden_layernodes))



class layer:
    def __init__(self,layer_nodesamt,layer_name,weight_amt):
        self.layer_nodesamt = layer_nodesamt
        self.layer_name = layer_name
        self.weight_amt = weight_amt
        self.layer_nodes = []
        self.create_nodes()

    def create_nodes(self):
        for x in range(self.layer_nodesamt):
            weight_array = []
            for y in range(self.weight_amt):
                weight_array.append(random.uniform(-4,4))
            self.layer_nodes.append(node(weight_array))
        #print(self.layer_nodes)


class node:
    def __init__(self,weights):
        # , inputs, weights, bias
        #self.i = inputs
        self.w = weights
        print(weights)
        self.b = random.uniform(-4,4)


    def NCalc(self):
        output = self.i[0]*self.w[0]+self.i[1]*self.w[1]+self.i[2]*self.w[2]+ self.b
        self.o = output



def create_network(hidden_layers, input_nodes, hidden_layernodes, output_nodes):
    the_network = network(hidden_layers, input_nodes, hidden_layernodes, output_nodes)






def main():
    # Versioner printes
    print("Python version:", sys.version)
    print("numpy version:", np.__version__)
    print("Matplotlib version:", matplotlib.__version__)
    inputs = [1.2,5.1,2.1]
    weights = [3.1,2.1,8.7]
    bias = 3
    #neuron = node(inputs,weights,bias)
    #neuron.NCalc()
    #print(neuron.o)
    create_network(HIDDEN_LAYERS,INPUT_NODES,HIDDEN_LAYERNODES,OUTPUT_NODES)




# Programmet kan kun køres som script, og kan ikke importeres i andre scripts.
if __name__ == "__main__":
    main()