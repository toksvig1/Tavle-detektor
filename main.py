# De nødvendige bibloteker bliver importeret her.
import numpy as np # Til matematik
import sys # Systemet
import matplotlib
import matplotlib.pyplot as plt # Til grafer
import random # Til weights og bias
import time # Til at finde ud af hvor lang tid programmet kørte
import math # Til Eulers tal
import pprint

start_time = time.time()
#np.random.seed(9150)

# Parametre ################################################

HIDDEN_LAYERS = 2
INPUT_NODES = 3
HIDDEN_LAYERNODES = 10
OUTPUT_NODES = 3
X = [[-1.2,-0.8,-2.4],[-1.4,-0.1,-1.7],[-0.6,-0.2,-0.7],[-1.2,-0.8,-2.6]]

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
        self.create_layersnp()
        
    def create_layersnp(self):
        for x in range(self.hidden_layersamt):
            if x == 0:
                self.hidden_layers.append(layer(self.input_nodes,self.hidden_layernodes))
            else:
                self.hidden_layers.append(layer(self.hidden_layernodes,self.hidden_layernodes))
        if self.hidden_layersamt > 0:
            self.output_layer.append(layer(self.hidden_layernodes,self.output_nodes))
        else:
            self.output_layer.append(layer(self.input_nodes,self.output_nodes))


    def forward_propagationnp(self,inputs,prediciton):
        for x in self.hidden_layers:
            #print("inputbf: "+str(self.maxim(x.layer_propagationnp(inputs))))
            #print("inputaf: "+str(np.maximum(0,x.layer_propagationnp(inputs))))



            #inputs = np.maximum(0,x.layer_propagationnp(inputs))
            inputs = self.maxim(x.layer_propagationnp(inputs))



            #print("inputs: "+str(inputs))
        self.result = self.output_layer[0].layer_propagationnp(inputs)
        self.result = self.softmax(self.result)
        print("Output: "+ str(self.result)) 
        zipped_list = list(zip(self.result, prediciton))
        loss_list = list(map(lambda x: -math.log(x[0][x[1]]),zipped_list))
        loss_mean = sum(loss_list)/len(loss_list)
        print("Loss: "+str(loss_mean))
        #xy = np.array(self.result).flatten()
        #print(len(xy))
        #xp = np.array(range(len(xy)))
        #plt.plot(xp,xy)
        #plt.show()
        
    def temp(self,inputs):
        for x in self.hidden_layers:
            inputs = self.maxim(x.layer_propagationnp(inputs))

        self.resultt = self.output_layer[0].layer_propagationnp(inputs)


    def maxim(self,inputs):
        rt = []
        for x in inputs:
            rtt = []
            for y in x:
                rtt.append(max(0,y))
            rt.append(rtt)
        return rt

    def softmax_maxim(self,out):
        ra = []
        for x in out:
            biggest_val = max(x)
            rta = list(map(lambda y:y-biggest_val,x))
            ra.append(rta)
        return ra

    def softmax(self,out):
        print(out)
        overflow_proc = self.softmax_maxim(out)
        e_xtemp = list(map(lambda nestedlist: list(map(lambda x: math.exp(x),nestedlist)),overflow_proc))

        e_x = np.exp(out-np.max(out, axis=1,keepdims=True))

        e_xsum = list(map(lambda x: sum(x),e_xtemp))
        zipe_x = list(zip(e_xtemp,e_xsum))
        
        
        e_xtempsum = list(map(lambda pair: list(map(lambda val: val/pair[1],pair[0])), zipe_x))
        #print("AAAA"+str(e_xtempsum))

        #print(zipe_x)
        #print("ASD"+str(e_x/np.sum(e_x, axis=1, keepdims=True)))
        #print("sammen: "+str(e_xtempsum))
        return e_xtempsum


    # SLETTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
    def create_layers(self):
        self.input_layer.append(layer(self.input_nodes,"input_layer",0))
        for x in range(self.hidden_layersamt):
            if x == 0:
                self.hidden_layers.append(layer(self.hidden_layernodes, "hidden_layer"+str(x+1),self.input_nodes))
            else:
                self.hidden_layers.append(layer(self.hidden_layernodes, "hidden_layer"+str(x+1),self.hidden_layernodes))
        if self.hidden_layersamt > 0:
            self.output_layer.append(layer(self.output_nodes, "output_layer",self.hidden_layernodes))
        else:
            self.output_layer.append(layer(self.output_nodes, "output_layer",self.input_nodes))
    # SLETTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
    def forward_propagation(self,inputs):
        self.input_layer[0].layer_propagation(inputs)
        for x in self.hidden_layers:
            if self.hidden_layers.index(x) == 0:
                x.layer_propagation(inputs)
            else:
                indx = self.hidden_layers.index(x)
                tmpinputs = self.hidden_layers[indx-1].get_all_outputs()
                print(tmpinputs)
                x.layer_propagation(tmpinputs)
        tmpinputs = self.hidden_layers[-1].get_all_outputs()
        self.output_layer[0].layer_propagation(tmpinputs)

class layer:
    def __init__(self,layer_nodesamt,layer_name):
        self.layer_nodesamt = layer_nodesamt
        self.layer_name = layer_name
        #self.weight_amt = weight_amt ,weight_amt
        self.layer_nodes = []

        #self.weights = np.random.randn(layer_nodesamt,layer_name)
        #print("----------------------")
        #print(layer_nodesamt)
        #print(layer_name)
        #print(self.weights)
        list_weights = [[random.uniform(-1,1) for _ in range(layer_name)] for _ in range(layer_nodesamt)]
        #pprint.pp(list_weights)
        #print("----------------------")
        self.weights = list_weights
        #self.biases = np.zeros((1, layer_name))
        biasespre = []
        biasespre.append([0 for x in range(layer_name)])
        self.biases = biasespre
        #print("WEIGHTS:")
        #print(self.weights)
        #print("BIASES")
        #print(self.biases)


        #self.create_nodes()


    def layer_propagationnp(self, inputs):
        #pprint.pp("inputs: "+str(inputs))
        #pprint.pp("weights: "+str(self.weights))
        weights_transposed = list(map(list, zip(*self.weights)))
        #pprint.pp("Weights transposed: "+str(list(map(list, zip(*self.weights)))))
        #pprint.pp("res: "+str(np.dot(inputs, self.weights)))
        #pprint.pp("weight : "+str(self.weights))
        #pprint.pp("weighwoit: "+str(weights_transposed))
        
        output = [[sum(x*y+self.biases[0][weights_transposed.index(a_row)] for x,y in zip(a_row, b_row)) for a_row in weights_transposed] for b_row in inputs]
        #pprint.pp("Test: "+str(output))
        self.layer_output = output
        #self.layer_output = np.dot(inputs, self.weights) + self.biases
        return self.layer_output


    # SLETTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
    def create_nodes(self):
        for x in range(self.layer_nodesamt):
            weight_array = []
            for y in range(self.weight_amt):
                weight_array.append(random.uniform(-4,4))
            self.layer_nodes.append(node(weight_array))
        #print(self.layer_nodes)
    # SLETTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
    def layer_propagation(self,inputs):
        print(self.layer_name)
        if self.layer_name != "input_layer":
            for x in self.layer_nodes:
                x.output_calculate(inputs)
        else:
            for x in self.layer_nodes:
                indx = self.layer_nodes.index(x)
                x.inputnode_output(inputs[indx])

    # SLETTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
    def get_all_outputs(self):
        output_to_input = []
        for x in self.layer_nodes:
            output_to_input.append(x.o)
        return output_to_input



# SLETTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
class node:
    def __init__(self,weights):
        # , inputs, weights, bias
        #self.i = inputs
        self.w = weights
        print("Weights: "+ str(weights))
        self.b = random.uniform(-4,4)
        print("Bias: "+ str(self.b))


    def output_calculate(self,inputs):
        # Slet kommentarer hvis klar
        # Finder outputtet af noden med skalarproduktet via numpy


        #output = 0
        #for x in inputs:
        #    indx = inputs.index(x)
        #    output = output + (x*self.w[indx])
        #output = output + self.b
        #output = self.i[0]*self.w[0]+self.i[1]*self.w[1]+self.i[2]*self.w[2]+ self.b
        self.o = np.dot(inputs,self.w)+self.b
        print("Node output: "+str(self.o))
        #print("Node ouput using dot: "+ str(np.dot(inputs,self.w)+self.b))

    def inputnode_output(self,input):
        self.o = input
        print("Input node input: "+str(self.o))

def create_network(hidden_layers, input_nodes, hidden_layernodes, output_nodes):
    return network(hidden_layers, input_nodes, hidden_layernodes, output_nodes)
    





def main():
    # Versioner printes
    print("Python version:", sys.version)
    print("numpy version:", np.__version__)
    print("Matplotlib version:", matplotlib.__version__)
    inputs = [1.2,5.1,2.1]
    weights = [3.1,2.1,8.7]
    bias = 3
    prediction = [0,1,2,0] 
    the_network = create_network(HIDDEN_LAYERS,INPUT_NODES,HIDDEN_LAYERNODES,OUTPUT_NODES)
    the_network.forward_propagationnp(X,prediction)
    print("")
    print("Process finished --- %s seconds ---" % (time.time() - start_time))
    print("Can run at %s fps." % (1/(time.time() - start_time)))


# Programmet kan kun køres som script, og kan ikke importeres i andre scripts.
if __name__ == "__main__":
    main()