from tkinter import *
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog as fd
import math
import time
import json
import main_fixed


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
    res = main_fixed.simulate_program(filenames,3,False,filename)
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


