import random
import getopt
from DLTools.Permutator import *
import sys,argparse
from numpy import arange
import os

# Input for Mixing Generator
FileSearch="/data/LArIAT/h5_files/*.h5"
#FileSearch="/Users/afarbin/LCD/Data/*/*.h5"

Particles= ['electron', 'antielectron',
            'pion',             
            'photon',
            'pionPlus', 'pionMinus',
            'proton', 'antiproton',
            'muon', 'antimuon',
            'kaonMinus', 'kaonPlus']

# Generation Model
Config={
    "MaxEvents":int(.5e6),
    "NTestSamples":25000,

    "Particles":Particles,
    "NClasses":len(Particles),

    "Epochs":1000,
    "BatchSize":128,

    "DownSampleSize":8,
    "ScanWindowSize":256,
    "Normalize":True,

    "EnergyCut":0.61,
    
    # Configures the parallel data generator that read the input.
    # These have been optimized by hand. Your system may have
    # more optimal configuration.
    "n_threads":50,  # Number of workers
    "n_threads_cache":4,  # Number of workers
    "multiplier":1, # Read N batches worth of data in each worker

    # How weights are initialized
    "WeightInitialization":"'normal'",


    # Model
    "View1":True,
    "View2":True,
    "Width":32,
    "Depth":2,

    # No specific reason to pick these. Needs study.
    # Note that the optimizer name should be the class name (https://keras.io/optimizers/)
    "loss":"'categorical_crossentropy'",

    "activation":"'relu'",
    "BatchNormLayers":True,
    "DropoutLayers":True,
    
    # Specify the optimizer class name as True (see: https://keras.io/optimizers/)
    # and parameters (using constructor keywords as parameter name).
    # Note if parameter is not specified, default values are used.
    "optimizer":"'RMSprop'",
    "lr":0.01,    
    "decay":0.001,

    # Parameter monitored by Callbacks
    "monitor":"'val_loss'",

    # Active Callbacks
    # Specify the CallBack class name as True (see: https://keras.io/callbacks/)
    # and parameters (using constructor keywords as parameter name,
    # with classname added).
    "ModelCheckpoint":True,
    "Model_Chekpoint_save_best_only":False,    

    # Configure Running time callback
    # Set RunningTime to a value to stop training after N seconds.
    "RunningTime": 2*3600,

    # Load last trained version of this model configuration. (based on Name var below)
    "LoadPreviousModel":True
}

# Parameters to scan and their scan points.
Params={    "optimizer":["'RMSprop'","'Adam'","'SGD'"],
            "Width":[32,64,128,256,512],
            "Depth":range(1,5),
            "lr":[0.01,0.001],
            "decay":[0.01,0.001],
          }

# Get all possible configurations.
PS=Permutator(Params)
Combos=PS.Permutations()
print "HyperParameter Scan: ", len(Combos), "possible combiniations."

# HyperParameter sets are numbered. You can iterate through them using
# the -s option followed by an integer .
i=0
if "HyperParamSet" in dir():
    i=int(HyperParamSet)

for k in Combos[i]: Config[k]=Combos[i][k]

# Build a name for the this configuration using the parameters we are
# scanning.
Name="LArTPCDNN"
for MetaData in Params.keys():
    val=str(Config[MetaData]).replace('"',"")
    Name+="_"+val.replace("'","")

if "HyperParamSet" in dir():
    print "______________________________________"
    print "ScanConfiguration"
    print "______________________________________"
    print "Picked combination: ",i
    print "Combo["+str(i)+"]="+str(Combos[i])
    print "Model Filename: ",Name
    print "______________________________________"
else:
    for ii,c in enumerate(Combos):
        print "Combo["+str(ii)+"]="+str(c)
