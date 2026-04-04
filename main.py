# De nødvendige bibloteker bliver importeret her.
import numpy as np # Til matematik
import sys # Systemet
import matplotlib
import matplotlib.pyplot as plt # Til grafer
import random # Til weights og bias
import time # Til at finde ud af hvor lang tid programmet kørte
import math # Til Eulers tal
import pprint
import json

start_time = time.time()
#np.random.seed(9150)

# Parametre ################################################

HIDDEN_LAYERS = 2
INPUT_NODES = 3
HIDDEN_LAYERNODES = 10
OUTPUT_NODES = 3
#X = [[-1.2,-0.8,-2.4],[-1.4,-0.1,-1.7],[-0.6,-0.2,-0.7],[-1.2,-0.8,-2.6]]
X = [[-1.2,-0.8,2.5],[-3.2,-2.8,1.5]]
# Parametre ################################################


# Det neurale netværk

class network:
    def __init__(self,hidden_layersamt, input_nodes, hidden_layernodes, output_nodes):
        self.hidden_layersamt = hidden_layersamt
        self.input_nodes = input_nodes
        self.hidden_layernodes = hidden_layernodes
        self.output_nodes = output_nodes
        self.hidden_layers = []
        self.layer_results_RELU = []
        self.input_layer = []
        self.output_layer = []
        self.create_layersnp()
        

    def load_file(self, load_file):
        with open(load_file, 'r') as file:
            data = json.load(file)
            self.hidden_layersamt = data['HIDDEN_LAYERS']
            self.input_nodes = data['INPUT_NODES']
            self.output_nodes = data['OUTPUT_NODES']
            self.hidden_layernodes = data['HIDDEN_LAYER_NODES']
            self.hidden_layers = []
            self.output_layer = []
            for x in range(self.hidden_layersamt):
                if x == 0:
                    self.hidden_layers.append(self.load_layer(self.input_nodes, self.hidden_layernodes))
                    self.hidden_layers[x].weights = data['HIDDEN_LAYER_WEIGHTS_'+str(x)]
                    self.hidden_layers[x].biases = data['HIDDEN_LAYER_BIAS_'+str(x)]
                else:
                    self.hidden_layers.append(self.load_layer(self.hidden_layernodes,self.hidden_layernodes))
                    self.hidden_layers[x].weights = data['HIDDEN_LAYER_WEIGHTS_'+str(x)]
                    self.hidden_layers[x].biases = data['HIDDEN_LAYER_BIAS_'+str(x)]
            if self.hidden_layersamt > 0:
                 self.output_layer.append(self.load_layer(self.hidden_layernodes, self.output_nodes))
                 self.output_layer[0].weights = data['OUTPUT_LAYER_WEIGHTS']
                 self.output_layer[0].biases = data['OUTPUT_LAYER_BIASES']

            else:
                 self.output_layer.append(self.load_layer(self.input_nodes, self.output_nodes))
                 self.output_layer[0].weights = data['OUTPUT_LAYER_WEIGHTS']
                 self.output_layer[0].biases = data['OUTPUT_LAYER_BIASES']
                


    def save_network(self):
        save_dict = {}
        save_dict['HIDDEN_LAYERS'] = self.hidden_layersamt
        save_dict['INPUT_NODES'] = self.input_nodes
        save_dict['OUTPUT_NODES'] = self.output_nodes
        save_dict['HIDDEN_LAYER_NODES'] = self.hidden_layernodes

        for x in self.hidden_layers:
            save_dict['HIDDEN_LAYER_WEIGHTS_'+str(self.hidden_layers.index(x))] = x.weights
            save_dict['HIDDEN_LAYER_BIAS_'+str(self.hidden_layers.index(x))] = x.biases

        save_dict['OUTPUT_LAYER_WEIGHTS'] = self.output_layer[0].weights
        save_dict['OUTPUT_LAYER_BIASES'] = self.output_layer[0].biases
        with open('network.json','w') as outfile:
            json.dump(save_dict, outfile)


    def load_layer(self, node_input, node_output):
        return layer(node_input, node_output)

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
            self.layer_results_RELU.append(inputs)


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
        

    def adam_optimization(self, softmax_output, skalar, inputs):
        parameters = {}
        for x in self.hidden_layers:
            node_weight_dict = x.weights
            node_bias_dict = x.biases
            parameters["dW"+str(self.hidden_layers.index(x)+1)] = node_weight_dict
            parameters["db"+str(self.hidden_layers.index(x)+1)] = node_bias_dict
        node_weight_dict = self.output_layer[0].weights
        node_bias_dict = self.output_layer[0].biases
        parameters["dW"+str(len(self.hidden_layers)+1)] = node_weight_dict
        parameters["db"+str(len(self.hidden_layers)+1)] = node_bias_dict
        w,b = self.batch_gradient(softmax_output, skalar)
        m, v = self.adam_initialization(parameters)
        for x in w[0]:
            w[0][w[0].index(x)] = [list(row) for row in zip(*x)]

    
        
        
        w[0] = [list(row) for row in zip(*w[0])]

        self.adam_update_parameters(parameters,w,b,m,v)
        

    def adam_update_parameters(self, parameters, w,b, m, v, t=1, learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        
        # For w, så er der 3 lister, hver liste har lister som tilhører hver deres batch: [[[b1],[b2],[b3],],[[b1],[b2],[b3]],[[b1],[b2],[b3]]]
        # bias er på samme måde, alle første lag er groupet sammen og resten er også groupet sammen.
        # de overstående går fra output til input, men selvfølgelig ikke inkluderende input.

        # m og v er bare dictionaries med hver weight og bias grupper for hver node er i deres egen liste.
        # de går fra input til output.
        # alt det samme tæller for parameters.

        # ----- I dette tilfælde bliver der ikke tage summen eller gennemsnittet af gradientsne.

        pprint.pp("parameters: "+str(parameters))
        pprint.pp("w: "+str(w))
        pprint.pp("b: "+str(b))
        pprint.pp("m: "+str(m))
        pprint.pp("v: "+str(v))
        
        # der regnes først for bias.
        # output til input
        for data_set in b:
            data_set_index = (len(b)-b.index(data_set))
            for batch in data_set:
                t = 1
                for data_point in batch:
                    corresponding_m = m['db'+str(data_set_index)][batch.index(data_point)]
                    corresponding_v = v['db'+str(data_set_index)][batch.index(data_point)]
                    m['db'+str(data_set_index)][batch.index(data_point)] = beta1*corresponding_m+(1-beta1)*data_point
                    v['db'+str(data_set_index)][batch.index(data_point)] = beta2*corresponding_v+(1-beta2)*(data_point**2)

                    m_corrected = m['db'+str(data_set_index)][batch.index(data_point)]/(1-beta1**t)
                    v_corrected = v['db'+str(data_set_index)][batch.index(data_point)]/(1-beta2**t)

                    parameters['db'+str(data_set_index)][batch.index(data_point)] = parameters['db'+str(data_set_index)][batch.index(data_point)]-learning_rate*m_corrected/(math.sqrt(v_corrected)+epsilon)
                    t += 1


        # Nu optimeres vægtene. Der gåes igennem hvert lag, igennem hver batch i de lag, igennem hvert datasæt i de lag
        # og til sidst igennem hver punkt i de datasæt.

        for data_set in w:
            data_set_index = (len(b)-b.index(data_set))



        #m = beta1*m + (1-beta1)*w
        #v = beta2*v + (1-beta2)*w**2

        #m_corrected = m/(1-beta1**t)
        #v_corrected = v/(1-beta2**t)

        #parameters = parameters-learning_rate*m_corrected / (v_corrected**(1/2)+epsilon)

        return parameters, m, v

    def batch_gradient(self, softmax_output, skalar):

        
        softmax_skalar_zip = list(zip(softmax_output,skalar))
        gradient_logits = list(map(lambda x: list(map(lambda y: y-1 if x[0].index(y) == x[1] else y, x[0])),softmax_skalar_zip))
        gradient_weights_zip = list(zip(self.layer_results_RELU[-1], gradient_logits))
        gradient_weights = [[[x * y for y in t[1]] for x in t[0]] for t in gradient_weights_zip]
        self.output_bias_gradient = gradient_logits
        self.output_weight_bias = gradient_weights

        #pprint.pp(gradient_logits)
        #pprint.pp("hidden out: "+str(self.layer_results_RELU[-1]))
        #pprint.pp("weights output: "+str(self.output_layer[0].weights))
        #pprint.pp("weights transposed: "+str([list(row) for row in zip(*self.output_layer[0].weights)]))
        #pprint.pp("resiutl: "+str(len(gradient_logits)))
        #pprint.pp("zipped : "+str(gradient_weights_zip))
        self.previous_error = gradient_logits
        self.hidden_layer_weight_gradients = []
        self.hidden_layer_bias_gradients = []
        self.hidden_layer_weight_gradients.append(self.output_weight_bias)
        self.hidden_layer_bias_gradients.append(self.output_bias_gradient)
        #pprint.pp("hidden weights: "+str(self.output_layer[0].weights))
        #pprint.pp("rwererewrew: "+str(gradient_logits))
        
        multiplied_weights = []
        for x in gradient_logits:
            sum_list = []
            for y in self.output_layer[0].weights:
                the_sum = 0
                for i in range(len(y)):
                    the_sum += y[i]*x[i]
                sum_list.append(the_sum)
            multiplied_weights.append(sum_list)
        #pprint.pp("multi: "+str(multiplied_weights))
        self.previous_error = multiplied_weights
        # måske ReLU
        # gang med input til sidste hidden layer
        #pprint.pp("inouts : "+str(self.hidden_layers[-1].layer_inputs))
        hidden_layer_one_weights = []
        for x in multiplied_weights:
            nested_list = []
            for b in x:
                maybe = []
                for y in self.hidden_layers[-1].layer_inputs[multiplied_weights.index(x)]:
                    #print("y "+str(multiplied_weights.index(x))+" "+str(y))
                    maybe.append(b*y)
                nested_list.append(maybe)
                #print("len : "+str(nested_list))
            hidden_layer_one_weights.append(nested_list)
        self.hidden_layer_weight_gradients.append(hidden_layer_one_weights)
        self.hidden_layer_bias_gradients.append(multiplied_weights)
        self.hidden_layer_one_weights = hidden_layer_one_weights
        self.hidden_layer_one_bias = multiplied_weights
        
        hidden_layers_iterable = self.hidden_layers
        hidden_layers_iterable.pop()
        hidden_layers_iterable.reverse()
        
        for hidden in hidden_layers_iterable:
            #pprint.pp("hidden weights: "+str(hidden.weights))
            multiplied_weights2 = []
            for x in self.previous_error:
                sum_list = []
                for y in [list(row) for row in zip(*hidden.weights)]:
                    the_sum = 0
                    for i in range(len(y)):
                        the_sum += y[i]*x[i]
                    sum_list.append(the_sum)
                multiplied_weights2.append(sum_list)
            #pprint.pp("multi: "+str(multiplied_weights2))
            self.previous_error = multiplied_weights2

            hidden_layer_one_weights = []
            for x in multiplied_weights2:
                nested_list = []
                for b in x:
                    maybe = []
                    for y in hidden.layer_inputs[multiplied_weights2.index(x)]:
                        #print("y "+str(multiplied_weights.index(x))+" "+str(y))
                        maybe.append(b*y)
                    nested_list.append(maybe)
                    #print("len : "+str(nested_list))
                hidden_layer_one_weights.append(nested_list)
            self.hidden_layer_weight_gradients.append(hidden_layer_one_weights)
            #pprint.pp(hidden_layer_one_weights)
            self.hidden_layer_bias_gradients.append(multiplied_weights2)
        

        return self.hidden_layer_weight_gradients, self.hidden_layer_bias_gradients

    def adam_initialization(self, parameters):
        dict_len = len(parameters) // 2
        v = {}
        s = {}

        for i in range(dict_len):
            s['dW'+str(i+1)] = list(map(lambda x: list(map(lambda xx: 0, x)), parameters['dW'+ str(i+1)]))
            s['db'+str(i+1)] = list(map(lambda x: list(map(lambda xx: 0,x)), parameters['db'+ str(i+1)]))
        v = s.copy()
        return v, s

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
        self.softmax_result = e_xtempsum
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
        self.layer_inputs = inputs
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
    prediction = [0,1,2,0] 
    the_network = create_network(HIDDEN_LAYERS,INPUT_NODES,HIDDEN_LAYERNODES,OUTPUT_NODES)
    the_network.load_file('network.json')
    the_network.forward_propagationnp(X,prediction)
    the_network.adam_optimization(the_network.softmax_result, prediction,X)
    #the_network.save_network()
    print("")
    print("Process finished --- %s seconds ---" % (time.time() - start_time))
    print("Can run at %s fps." % (1/(time.time() - start_time)))


# Programmet kan kun køres som script, og kan ikke importeres i andre scripts.
if __name__ == "__main__":
    main()