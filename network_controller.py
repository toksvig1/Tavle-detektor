# De nødvendige bibloteker bliver importeret her.
import numpy as np # Til matematik
import sys # Systemet
from PIL import Image # Til at indlæse billedet
from pathlib import Path # Til fil-paths
import matplotlib
import matplotlib.pyplot as plt # Til grafer
import random # Til weights og bias
import time # Til at finde ud af hvor lang tid programmet kørte
import math # Til Eulers tal
import json
import os



# Parametre ################################################

#HIDDEN_LAYERS = 2
#INPUT_NODES = 1024
#HIDDEN_LAYERNODES = 128
#OUTPUT_NODES = 3
#X = [[-1.2,0.8,-2.4],[-1.4,-0.1,1.7],[-0.6,-0.2,-0.7],[-1.2,0.8,-2.6]]


sign_names = ["20 sign","30 sign","50 sign","60 sign","70 sign","80 sign","80 ended sign","100 sign","120 sign","oncomming traffic"]

# Parametre ################################################


# Det neurale netværk

class network:
    def __init__(self,hidden_layersamt, input_nodes, hidden_layernodes, output_nodes):
        # Skaber alle lag i netværket.

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
        # Indlæser det gemte netværk fra en JSON fil.

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
        # Gemmer netværkets weights, biases og diverse værdier, til en JSON fil.

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
        # Bruges til at opdatere lag, med de gemte værdier til netværket.
        return layer(node_input, node_output)

    def create_layersnp(self):
        # Skaber alle lag. De lægges ind i attributter, der gemmer dem.
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
        # Normal forward propagation, dog uden numpy.

        self.layer_results_RELU = []
        for x in self.hidden_layers:
            inputs = self.maxim(x.layer_propagationnp(inputs))
            self.layer_results_RELU.append(inputs)

        self.result = self.output_layer[0].layer_propagationnp(inputs)
        self.result = self.softmax(self.result)
        zipped_list = list(zip(self.result, prediciton))
        self.predicloss = zipped_list
        loss_list = list(map(lambda x: -math.log(max(x[0][x[1]], 1e-15)), zipped_list))
        acc = []
        for x in zipped_list:
            max_val = max(x[0])
            max_val_indx = x[0].index(max_val)
            if max_val_indx == x[1]:
                acc.append(100)
            else:
                acc.append(0)
        self.accuracy = sum(acc)/len(acc)

        loss_mean = sum(loss_list)/len(loss_list)
        self.loss = loss_mean


    def init_adam(self):
        # Skaber begge momenter.
        # Begge har nul værdier i stedet for rigtige momenter, da det er første iteration.
        self.m = []
        self.v = []

        all_layers = self.hidden_layers + self.output_layer

        for lay in all_layers:
            m_w = [[0 for _ in row] for row in lay.weights]
            v_w = [[0 for _ in row] for row in lay.weights]

            m_b = [[0 for _ in lay.biases[0]]]
            v_b = [[0 for _ in lay.biases[0]]]

            self.m.append({'w': m_w, 'b': m_b})
            self.v.append({'w': v_w, 'b': v_b})

        self.t = 0


    def batch_gradient(self, softmax_output, skalar):
        # Beregner gradients for alle weights og biases.
        #
        batch_size = len(softmax_output)
        layer_count = len(self.hidden_layers) + 1


        # bruger list comprehension til at skabe listen til gradientsene
        dW = [None for _ in range(layer_count)]
        dB = [None for _ in range(layer_count)]

        # ---------- output lag ----------
        # starter med det sidste lag
        output_delta = []
        for sample_index in range(batch_size):
            probs = softmax_output[sample_index][:]
            probs[skalar[sample_index]] -= 1
            output_delta.append(probs)

        # tildeler den tidligere error loss baseret på mængden af hidden layers 
        if len(self.hidden_layers) > 0:
            prev_activation = self.layer_results_RELU[-1]
        else:
            prev_activation = self.output_layer[0].layer_inputs


        # Regner gradientsene ved at gange dem med deres respektive inputs
        output_weight_grad = []
        for input_index in range(len(prev_activation[0])):
            row = []
            for output_node_index in range(len(output_delta[0])):
                s = 0
                for sample_index in range(batch_size):
                    s += prev_activation[sample_index][input_index] * output_delta[sample_index][output_node_index]
                row.append(s / batch_size)
            output_weight_grad.append(row)


        # finder bias gradientene med samme metode
        output_bias_grad = []
        bias_row = []
        for output_node_index in range(len(output_delta[0])):
            s = 0
            for sample_index in range(batch_size):
                s += output_delta[sample_index][output_node_index]
            bias_row.append(s / batch_size)
        output_bias_grad.append(bias_row)

        # tilføjer den til de tidligere skabte lister
        dW[layer_count - 1] = output_weight_grad
        dB[layer_count - 1] = output_bias_grad

        # Går igennem alle hidden layers
        # De tidligere gradients var kun for output laget
        next_delta = output_delta

        # Går fra input mod output
        if len(self.hidden_layers) > 0:
            next_weights = self.output_layer[0].weights

            for hidden_index in range(len(self.hidden_layers) - 1, -1, -1):
                current_layer = self.hidden_layers[hidden_index]
                current_delta = []

                for sample_index in range(batch_size):
                    sample_delta = []

                    for node_index in range(len(current_layer.biases[0])):
                        s = 0
                        for next_node_index in range(len(next_delta[0])):
                            s += next_delta[sample_index][next_node_index] * next_weights[node_index][next_node_index]
                        relu_grad = 1 if self.layer_results_RELU[hidden_index][sample_index][node_index] > 0 else 0
                        sample_delta.append(s * relu_grad)

                    current_delta.append(sample_delta)
                # Bestemmer hvilke error loss der bruges
                if hidden_index == 0:
                    prev_activation = current_layer.layer_inputs
                else:
                    prev_activation = self.layer_results_RELU[hidden_index - 1]
                # Kører det sidste lag
                weight_grad = []
                for input_index in range(len(prev_activation[0])):
                    row = []
                    for node_index in range(len(current_delta[0])):
                        s = 0
                        for sample_index in range(batch_size):
                            s += prev_activation[sample_index][input_index] * current_delta[sample_index][node_index]
                        row.append(s / batch_size)
                    weight_grad.append(row)

                bias_grad = []
                bias_row = []
                for node_index in range(len(current_delta[0])):
                    s = 0
                    for sample_index in range(batch_size):
                        s += current_delta[sample_index][node_index]
                    bias_row.append(s / batch_size)
                bias_grad.append(bias_row)

                dW[hidden_index] = weight_grad
                dB[hidden_index] = bias_grad

                next_delta = current_delta
                next_weights = current_layer.weights

        return dW, dB


    def adam_step(self, w, b, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        # Adam step funktionen.
        # Træner netværket en gang, baseret på det tidligere resultat.


        self.t += 1
        all_layers = self.hidden_layers + self.output_layer

        for layer_index, lay in enumerate(all_layers):
            # weights
            for i in range(len(lay.weights)):
                for j in range(len(lay.weights[i])):
                    grad = w[layer_index][i][j]

                    self.m[layer_index]['w'][i][j] = beta1 * self.m[layer_index]['w'][i][j] + (1 - beta1) * grad
                    self.v[layer_index]['w'][i][j] = beta2 * self.v[layer_index]['w'][i][j] + (1 - beta2) * (grad ** 2)

                    m_hat = self.m[layer_index]['w'][i][j] / (1 - beta1 ** self.t)
                    v_hat = self.v[layer_index]['w'][i][j] / (1 - beta2 ** self.t)

                    lay.weights[i][j] -= lr * m_hat / (math.sqrt(v_hat) + eps)

            # biases
            for j in range(len(lay.biases[0])):
                grad = b[layer_index][0][j]

                self.m[layer_index]['b'][0][j] = beta1 * self.m[layer_index]['b'][0][j] + (1 - beta1) * grad
                self.v[layer_index]['b'][0][j] = beta2 * self.v[layer_index]['b'][0][j] + (1 - beta2) * (grad ** 2)

                m_hat = self.m[layer_index]['b'][0][j] / (1 - beta1 ** self.t)
                v_hat = self.v[layer_index]['b'][0][j] / (1 - beta2 ** self.t)

                lay.biases[0][j] -= lr * m_hat / (math.sqrt(v_hat) + eps)


    def adam_optimization(self, iterations, epoch_amt,class_range,batch_size):
        # Adam optimering
        # En batch af 'batch_size' billeder, bliver kørt igennem, og bagefter køres
        # adam_step() for at lave de relevante beregninger til at opdatere weights og biases.


        self.init_adam()
        self.loss_graph_data = []
        self.acc_graph_data = []

        for itera in range(iterations):
            ent_acc = []
            for epoch in range(epoch_amt):
                inputs2, skalar2 = gather_input(r"C:\Users\htkda\Downloads\Dataset\Dataset",batch_size,class_range)
                self.forward_propagationnp(inputs2, skalar2)
                w, b = self.batch_gradient(self.softmax_result, skalar2)
                self.adam_step(w, b)
                ent_acc.append(self.accuracy)
                self.loss_graph_data.append(self.loss)
                self.acc_graph_data.append(self.accuracy)

            self.forward_propagationnp(inputs2, skalar2)
            print("---------------------  " + str((itera + 1) * 100) + "  ---------------------")
            print("New loss: " + str(self.loss))
            print("Accuracy of the model: "+str(sum(ent_acc)/len(ent_acc)))
            self.model_accuracy = sum(ent_acc)/len(ent_acc)
        plt.plot(list(range(len(self.loss_graph_data))), self.loss_graph_data, label = "Loss curve")
        plt.plot(list(range(len(self.loss_graph_data))), self.acc_graph_data, label = "Accuracy curve")
        plt.legend()
        plt.show()

    def temp(self,inputs):
        for x in self.hidden_layers:
            inputs = self.maxim(x.layer_propagationnp(inputs))

        self.resultt = self.output_layer[0].layer_propagationnp(inputs)


    def maxim(self,inputs):
        # Fungerer som np.max(). 
        # ReLU aktivations funktionen.
        rt = []
        for x in inputs:
            rtt = []
            for y in x:
                rtt.append(max(0,y))
            rt.append(rtt)
        return rt

    def softmax_maxim(self,out):
        # Overflow beskyttelse til softmax funktionen.
        ra = []
        for x in out:
            biggest_val = max(x)
            rta = list(map(lambda y:y-biggest_val,x))
            ra.append(rta)
        return ra

    def softmax(self,out):
        # Softmax aktivation, bruges til det sidste lag. 
        # Eulers tal opløftes i alle værdierne af udgangslaget. 
        # Hvert tal divideres af summen af alle tal.

        overflow_proc = self.softmax_maxim(out)
        e_xtemp = list(map(lambda nestedlist: list(map(lambda x: math.exp(x),nestedlist)),overflow_proc))

        e_xsum = list(map(lambda x: sum(x),e_xtemp))
        zipe_x = list(zip(e_xtemp,e_xsum))

        e_xtempsum = list(map(lambda pair: list(map(lambda val: val/pair[1],pair[0])), zipe_x))
        self.softmax_result = e_xtempsum
        return e_xtempsum


class layer:
    def __init__(self,layer_nodesamt,layer_name):
        # Alle attributter får deres værdier tildelt.
        # Vægtene skabes. Deres værdier går fra -1 til 1.
        self.layer_nodesamt = layer_nodesamt
        self.layer_name = layer_name
        self.layer_nodes = []

        list_weights = [[random.uniform(-1,1) for _ in range(layer_name)] for _ in range(layer_nodesamt)]
        self.weights = list_weights
        biasespre = []
        biasespre.append([0 for x in range(layer_name)])
        self.biases = biasespre


    def layer_propagationnp(self, inputs):
        # Forward propagation. Vægtene tranposes, så dimensionerne passer.
        # Der bruges list comprehension til at regne resultaterne. 
        # Regnes som skalar produktet.
        self.layer_inputs = inputs
        weights_transposed = list(map(list, zip(*self.weights)))
        output = [[sum(x*y+self.biases[0][weights_transposed.index(a_row)] for x,y in zip(a_row, b_row)) for a_row in weights_transposed] for b_row in inputs]
        self.layer_output = output
        return self.layer_output


def create_network(hidden_layers, input_nodes, hidden_layernodes, output_nodes):
    return network(hidden_layers, input_nodes, hidden_layernodes, output_nodes)

def train_init(folder, HIDDEN_LAYERS, INPUT_NODES, HIDDEN_LAYERNODES, OUTPUT_NODES,epoch_amt,batch_amt,class_range, batch_size):
    #   Netværket skabes med de givne parametre. 
    #   ReLU activation, softmax activation, cross entropy og adam optimizing.
    print("Python version:", sys.version)
    print("Matplotlib version:", matplotlib.__version__)
    start_time = time.time()
    the_network = create_network(HIDDEN_LAYERS,INPUT_NODES,HIDDEN_LAYERNODES,OUTPUT_NODES)
    first_batch, first_prediction = gather_input(folder,1,class_range)
    the_network.forward_propagationnp(first_batch,first_prediction)
    the_network.adam_optimization(batch_amt, epoch_amt, class_range, batch_size)
    #the_network.save_network()

    print("Process finished --- %s seconds ---" % (time.time() - start_time))

    return the_network.model_accuracy

    


def gather_input(folder,batch_size,class_range):
    # Brugt at netværket til at indsamle data på billederne.
    # Dataet bliver brugt til at øve netværket.
    input_batches = []
    skalar_prediction = []
    for batch in range(batch_size):
        class_type = random.randint(0,class_range-1)
        skalar_prediction.append(class_type)
        path = Path(folder+"/"+str(class_type))
        images = os.listdir(path)
        the_image = images[random.randint(0,len(images)-1)]
        the_image_grayscale = Image.open(folder+"/"+str(class_type)+"/"+the_image).convert('L')
        the_image_data = list(the_image_grayscale.getdata())
        the_image_data = [the_image_data[offset:offset+32] for offset in range(0, 32*32, 32)]
        the_image_data = [item for sublist in the_image_data for item in sublist]
        input_batches.append(the_image_data)
    return input_batches, skalar_prediction

def input_of_single_image(image_path,class_range):
    # Brugt at network_runner.py til at få dataet af et enkelt billede.
    #
    input_batch = []
    skalar_prediction = [class_range]
    path = image_path
    the_image_grayscale = Image.open(path).convert('L')
    the_image_data = list(the_image_grayscale.getdata())
    the_image_data = [the_image_data[offset:offset+32] for offset in range(0, 32*32, 32)]
    the_image_data = [item for sublist in the_image_data for item in sublist]
    input_batch.append(the_image_data)

    return input_batch, skalar_prediction


def simulate_program(folder, class_range, singleOrMultiple, specific_image, HIDDEN_LAYERS, INPUT_NODES, HIDDEN_LAYERNODES, OUTPUT_NODES):
    # Funktionen der bruges af network_runner.py, til at kører netværket.
    #
    the_network = create_network(HIDDEN_LAYERS,INPUT_NODES,HIDDEN_LAYERNODES,OUTPUT_NODES)
    the_network.load_file(folder)


    
    if singleOrMultiple == True:
        start_time = time.time()
        predictionvals = []
        test_length = 100
        x = 0
        while x<test_length:
            time.sleep(5)
            input_image, prediction = gather_input(folder,1,class_range)
            the_network.forward_propagationnp(input_image,prediction)
            print("Færdig med billede: "+str(x))
            res = max(the_network.softmax_result[0])
            res_index=(the_network.softmax_result[0].index(res))
            if prediction[0] == res_index:
                predictionvals.append(100)
            else:
                predictionvals.append(0)
            x +=1
        print("Accuracy :"+str(sum(predictionvals)/len(predictionvals)))
        print("")
        print("Process finished --- %s seconds ---" % (time.time() - start_time))
        return sum(predictionvals)/len(predictionvals)
    else:
        batch, skalar = input_of_single_image(specific_image,1)
        start_time = time.time()
        the_network.forward_propagationnp(batch,skalar)
        print("Process finished --- %s seconds ---" % (time.time() - start_time))
        print("Can run at %s fps." % (1/(time.time() - start_time)))
        res = max(the_network.softmax_result[0])
        res_index=(the_network.softmax_result[0].index(res))
        print("The network predicts that this image, is a: "+sign_names[res_index])
        return sign_names[res_index]


def main():
    # Funktion der kører, hvis denne fil bliver kørt alene.

    print(" ")
    print(" ")
    print("-----------------------------------------")
    print("Dette program blev lavet i forbindelse med et programmering eksamensprojekt.")
    print("-----------------------------------------")
    print("Denne fil kan ikke køres alene.")
    print("Brug network_trainer.py eller network_runner.py, til at køre netværket.")
    print(" ")
    print(" ")

# Programmet kører kun main(), hvis det køres som hoved programmet.
if __name__ == "__main__":
    main()
