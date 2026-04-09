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

root.minsize(480, 190)
root.maxsize(480, 190)

root.rowconfigure(0,weight=1)
root.rowconfigure(1,weight=1)
root.rowconfigure(2,weight=1)
root.rowconfigure(3,weight=1)
root.rowconfigure(4,weight=1)
root.rowconfigure(5,weight=1)
root.rowconfigure(6,weight=1)
root.rowconfigure(7,weight=1)
root.rowconfigure(8,weight=1)

root.columnconfigure(0,weight=1)

def select_json():
    filetypes = (('JSON files','*.json'),)
    global filenames
    filenames = fd.askdirectory()
    
    print(filenames)

def trainer():
    HIDDEN_LAYERS = int(text3.get('1.0', tk.END))
    INPUT_NODES = int(text4.get('1.0', tk.END))
    HIDDEN_LAYERNODES = int(text5.get('1.0', tk.END))
    OUTPUT_NODES = int(text7.get('1.0', tk.END))
    epoch_amt = int(text1.get('1.0', tk.END))
    batch_amt = int(text2.get('1.0', tk.END))
    class_range = OUTPUT_NODES
    batch_size = int(text6.get('1.0', tk.END))

    



    label8 = Label(root,text='Netværket træner... Dette kan tage nogle minutter til nogle timer...').grid(column=0,row=8,sticky=W)
    acc = main_fixed.train_init(filenames,HIDDEN_LAYERS,INPUT_NODES,HIDDEN_LAYERNODES,OUTPUT_NODES,epoch_amt,batch_amt,class_range,batch_size)
    label8 = Label(root,text='Resultat: '+str(acc)+"%                                                                                                                  ").grid(column=0,row=8,sticky=W)


json_button = ttk.Button(
    root,
    text='Indlæs JSON fil',
    command=select_json
).grid(column=0,row=0,sticky=W)

train_button = ttk.Button(
    root,
    text='Træn netværkket',
    command=trainer
).grid(column=0,row=0,sticky=E)

label1 = Label(root,text='Mængden af epochs: ').grid(column=0,row=1,sticky=W)
label2 = Label(root,text='Mængden af træningsrunder: ').grid(column=0,row=2,sticky=W)
label3 = Label(root,text='Mængden af gemte lag: ').grid(column=0,row=3,sticky=W)
label4 = Label(root,text='Mængden af inputsnoder: ').grid(column=0,row=4,sticky=W)
label5 = Label(root,text='Mængden af noder i gemte lag: ').grid(column=0,row=5,sticky=W)
label6 = Label(root,text='Batch størrelse: ').grid(column=0,row=6,sticky=W)
label7 = Label(root,text='Mængden af udgangsnoder: ').grid(column=0,row=7,sticky=W)
label8 = Label(root,text='Resultat: ').grid(column=0,row=8,sticky=W)
# Text felter
text1 = tk.Text(root, height=0.15)
text1.grid(column=0,row=1,padx=200)
text2 = tk.Text(root, height=0.15)
text2.grid(column=0,row=2,padx=200)
text3 = tk.Text(root, height=0.15)
text3.grid(column=0,row=3,padx=200)
text4 = tk.Text(root, height=0.15)
text4.grid(column=0,row=4,padx=200)
text5 = tk.Text(root, height=0.15)
text5.grid(column=0,row=5,padx=200)
text6 = tk.Text(root, height=0.15)
text6.grid(column=0,row=6,padx=200)
text7 = tk.Text(root, height=0.15)
text7.grid(column=0,row=7,padx=200)

root.mainloop()