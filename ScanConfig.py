import random
import getopt
from DLTools.Permutator import *
import sys,argparse

Name="EventClassificationDNN"

Config={

    "MaxEvents":50000,
    "Epochs":1000,
    "BatchSize":2048*8,
    
    "LearningRate":0.005,
    
    "Decay":0.,
    "Momentum":0.,
    "Nesterov":0.,

    "WeightInitialization":"'normal'"
}

Params={ "Width":[128],
         "Depth":[2],
         "loss":[#"'mean_squared_error'",  
                 '"categorical_crossentropy"'],
         }

PS=Permutator(Params)
Combos=PS.Permutations()

print "HyperParameter Scan: ", len(Combos), "possible combiniations."

if "HyperParamSet" in dir():
    i=int(HyperParamSet)
else:
    # Set Seed based on time
    random.seed()
    i=int(round(len(Combos)*random.random()))
    print "Randomly picking HyperParameter Set"
    

if i<0:
    print "SetList:"
    for j in xrange(0,len(Combos)):
        print j,":",Combos[j]

    quit()


print "Picked combination: ",i

for k in Combos[i]:
    Config[k]=Combos[i][k]

for MetaData in Params.keys():
    val=str(Config[MetaData]).replace('"',"")
    Name+="_"+val.replace("'","")

print "Model Filename: ",Name


