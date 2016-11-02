import h5py
import glob
import numpy as np
from MultiClassData import *
from time import time


#[u'Eng', u'Track_length', u'enu_truth', u'features', u'lep_mom_truth', u'mode_truth', u'pdg']

#Samples = [ "electron", "pion_0"]

#Nx=2500,Ny=2,Nz=240,Nw=4096
def DownSample(y,factor,Nx,Ny,Nz,Nw,sumabs=False):
    if factor==0:
        return np.reshape(y,[Nx,Ny,Nz,Nw]),Nw
    # Remove entries at the end so Down Sampling works
    NwNew=Nw-Nw%factor
    features1=np.reshape(y,[Nx,Ny,Nz,Nw])[:,:,:,0:NwNew]
    # DownSample
    if sumabs:
       features_Down=abs(features1.reshape([Nz*NwNew/factor,factor])).sum(axis=3).reshape([Nx,Ny,Nz,NwNew/factor])
    else:
        features_Down=features1.reshape([Nx,Ny,Nz*NwNew/factor,factor]).sum(axis=3).reshape([Nx,Ny,Nz,NwNew/factor])
    return features_Down, NwNew




def shuffle_in_unison_inplace(a, b, c=False):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    if type(c) != bool:
        return a[p], b[p], c[p]
    return a[p], b[p]

Samples = [ "electron", "pion_0"]

def LArIATDataGenerator(InSamples,FileSearch="/data/LArIAT/h5_files"):

    Files={}
    NEvents={}
    ClassIndex={}
    NSamples=len(InSamples)
    Status={}

    for I,Sample in enumerate(InSamples):
        Files[Sample]=glob.glob(FileSearch+"/*"+Sample+"*.h5")
        NEvents_File=0
        ClassIndex[Sample]=I
        print "ClassIndex   =  ",ClassIndex
        Status[Sample]= { "fileI":0,"eventI":0}
    
    while True:
        
        First=True
        FileCount=0
        StartI=0
        NRead={}
        
        for Sample in Files:
            NRead[Sample] = 500
        
        for Sample in Files:

            FileCount+=1
            ParticleName=Sample
            FileName=Files[Sample][Status[Sample]["fileI"]]
            print "Opening File:",FileName
            f=h5py.File(FileName,"r")
            NEvents_File=f["features"].shape[0]
            print "NEvent_File   ",NEvents_File
            StartI=Status[Sample]["eventI"]
            print "Firt:StartI",StartI

            if NEvents_File < 2500:
                StartI+=StartI
                print "Second StartI",StartI
            else:
                Status[Sample]["eventI"]=0
                Status[Sample]["fileI"]+=1
                print "Files changed to next"
            
            y = f["features"]
            Nx = NEvents_File
            Ny = 2
            Nz = 240
            Nw = 4096
            Factor = 3
            features_Down,NwNew=DownSample(y,Factor,Nx,Ny,Nz,Nw)
            print "features_Down shape",features_Down.shape
            TheShape= features_Down.shape[:]
            if First:

               Data_X=np.zeros(TheShape)
               Data_Y=np.zeros((NEvents_File))
               First = False

            Data_X[:]= features_Down[:]
            print "Data_X shape",Data_X.shape
            a=np.empty(NEvents_File); a.fill(ClassIndex[Sample])
            print "shape a",a.shape, "np.array a",np.array(a)
            Data_Y[:]= a
            f.close()
        print "FileCount",FileCount

    Data_X,Data_Y=shuffle_in_unison_inplace(Data_X,Data_Y)
    yield (Data_X,Data_Y),ClassIndex,NEvents_File
                 

#if MaxFiles>0:
#if FileCount>MaxFiles:
#break


xx = LArIATDataGenerator(Samples)

I=0
for (x,y),ClassIndex,NEvents_File in xx:
    print x.shape,y.shape,ClassIndex,"NEvents_File  ",NEvents_File
    if I >5:
         break
    I +=1


