from tkinter import *
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog as fd
import math
import time
import json
import network_controller


root = Tk()
root.title("Netværk kører")

root.minsize(480, 120)
root.maxsize(480, 120)

root.rowconfigure(0,weight=1)
root.rowconfigure(1,weight=1)
root.rowconfigure(2,weight=2)

root.columnconfigure(0,weight=1)


def select_json():
    filetypes = (('JSON files','*.json'),)
    global filenames
    filenames = fd.askopenfilename(
        title='Indlæs JSON fil',
        initialdir='/',
        filetypes=filetypes)
    
    print(filenames)


def select_image():
    filetypes = (('JSON files','*.png'),)
    global filename
    filename = fd.askopenfilename(
        title='Vælg billede',
        initialdir='/',
        filetypes=filetypes)
    
    print(filename)
    

def run_ai():
    HIDDEN_LAYERS = 0
    HIDDEN_LAYERNODES = 0
    INPUT_NODES = 0
    OUTPUT_NODES = 0
    with open(filenames, 'r') as file:
            data = json.load(file)
            HIDDEN_LAYERS = data['HIDDEN_LAYERS']
            HIDDEN_LAYERNODES = data['HIDDEN_LAYER_NODES']
            INPUT_NODES = data['INPUT_NODES']
            OUTPUT_NODES = data['OUTPUT_NODES']

    res = network_controller.simulate_program(filenames,3,False,filename,HIDDEN_LAYERS, INPUT_NODES, HIDDEN_LAYERNODES, OUTPUT_NODES)
    label1 = Label(root,text='Resultat: '+res).grid(column=0,row=1,sticky=W)
    print(res)
    
json_button = ttk.Button(
    root,
    text='Indlæs JSON fil',
    command=select_json
)
image_button = ttk.Button(
    root,
    text='Vælg billede',
    command=select_image
)

test_button = ttk.Button(
    root,
    text='Kør billede',
    command=run_ai
)

label1 = Label(root,text='Resultat: ').grid(column=0,row=1,sticky=W)

json_button.grid(column=0,row=0,sticky=W)
image_button.grid(column=0,row=0)
test_button.grid(column=0,row=0,sticky=E)

root.mainloop()


