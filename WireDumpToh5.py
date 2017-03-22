
import json

from tables import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('inFileName')
parser.add_argument('outFileName')
args = parser.parse_args()

inFileName=args.inFileName
h5FileName=args.outFileName

print "Input file: ",inFileName
print "Output file: ",h5FileName

VariableLength=True

if VariableLength:
    print "Writing Variable Length output."
else:
    print "Writing Fixed Length output."

#Flush table every 100 events... in table mode.
FlushCount=128

h5file = open_file(h5FileName, mode = "w", title = "Test file")
FILTERS = Filters(complib='zlib', complevel=5)
group = h5file.create_group("/", 'Events', 'Events')

class Event:
    def __init__(self,Schema,file,root,FILTERS, Name="MCTrajectories", VariableLength=True, postfix="Val"):
        self.Schema=Schema
        self.VariableLength=VariableLength
        self.postfix=postfix
        
        for k in Schema:
            if VariableLength:
                setattr(self,k,file.create_vlarray(root,k,Schema[k],"",filters=FILTERS))
            setattr(self,k+self.postfix,[])

        if not VariableLength:
            self.table = h5file.create_table(root, Name, Schema, "",filters=FILTERS)

        self.i=0

    def clear(self):
        for k in Schema:
            setattr(self,k+self.postfix,[])


    def addInstance(self,Data):
        for k in Data:
            try:
                getattr(self,k+self.postfix).append(Data[k])
#                print k,Data[k]
            except:
#                print k, "Not in schema."
                pass

    def Fill(self):
        self.i+=1
        if self.VariableLength:
            for k in self.Schema:
                getattr(self,k).append(getattr(self,k+self.postfix))
                print self.i,k
                #setattr(self,k+self.postfix,[])
        else:
            anEvent=self.table.row

            for k in self.Schema:
#                print k,getattr(self,k+self.postfix)
                try:
                    anEvent[k]=getattr(self,k+self.postfix)[0]
                except:
#                    print k,getattr(self,k+self.postfix)
                    pass
                setattr(self,k+self.postfix,[])

            anEvent.append()
            if self.i%FlushCount==0:
                self.table.flush()

            
Schema1={
    "X" : Float64Atom(shape=()),
    "Y" : Float64Atom(shape=()),
    "Z" : Float64Atom(shape=()),
    "E" : Float64Atom(shape=()),
}

Schema2={
    "X" : Float64Col(shape=()),
    "Y" : Float64Col(shape=()),
    "Z" : Float64Col(shape=()),
    "E" : Float64Col(shape=()),
}


if VariableLength:
    Schema=Schema1
else:
    Schema=Schema2

MyEventModel=Event(Schema,h5file,group,FILTERS,VariableLength)

import ROOT
import numpy as np


#"/home/sshahsav/3D_test_root_file/WireDump_electron_MC_Track_Shower.root"
f=ROOT.TFile(inFileName)
t=f.Get("wiredump/anatree")

for Event_I in xrange(0,t.GetEntries()):
    t.GetEntry(Event_I)
    MyEventModel.clear()
    Px=np.array(t.MidPx)
    Py=np.array(t.MidPy)
    Pz=np.array(t.MidPz)

    X=np.array(t.MidPosX)
    Y=np.array(t.MidPosY)
    Z=np.array(t.MidPosZ)

    E=np.sqrt(Px**2+Py**2+Pz**2)
    E1=E[1:]
    DeltaE=E[:-1]-E1

    NT=t.NTrTrajPts[0]-1

    #    for Traj_I in xrange(0,NT):
    MyEventModel.X.append(Px[:NT])#[Traj_I]
    MyEventModel.Y.append(Py[:NT])#[Traj_I]
    MyEventModel.Z.append(Pz[:NT])#[Traj_I]
    MyEventModel.E.append(E[:NT])#[Traj_I])

    #MyEventModel.Fill()
                            
f.Close()


#from scipy.sparse import coo_matrix
#import matplotlib.pyplot as plt
#import plotly.plotly as py
#from mpl_toolkits.mplot3d import Axes3D

# h=np.histogramdd(np.array([Px,Py,Pz]).transpose(),bins=[100,100,100],weights=E)

# # Color Map... convert E to color values
# #http://matplotlib.org/users/colormapnorms.html

# NT=t.NTrTrajPts[0]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(Px[0:NT],Py[0:NT],Pz[0:NT],c=E[0:NT])
# fig.show()
