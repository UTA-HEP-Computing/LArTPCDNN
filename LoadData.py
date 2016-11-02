import h5py
import glob
import numpy as np
from MultiClassData import *
from time import time


#[u'Eng', u'Track_length', u'enu_truth', u'features', u'lep_mom_truth', u'mode_truth', u'pdg']

Samples = [ "electron", "pion_0"]

def shuffle_in_unison_inplace(a, b, c=False):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    if type(c) != bool:
        return a[p], b[p], c[p]
    return a[p], b[p]

def DataGenerator(Samples,BatchSize=1024,path="/data/LArIAT/h5_files"):
    Files={}
    NEvents={}
    ClassIndex={}
    
    NSamples=len(Samples)
    Status={}
    
    for I,Sample in enumerate(Samples):
        Files[Sample]=glob.glob(path+"/*"+Sample+"*.h5")
        NEvents[Sample]=0
        ClassIndex[Sample]=I
        Status[Sample]= { "fileI":0,
                          "eventI":0}

    while True:
        start=time()
        First=True
        I=0
        NRead={}
        for Sample in Files:
            NRead[Sample]=BatchSize/NSamples
        for Sample in Files:
            NTotal=0
            while NTotal<NRead[Sample]:
                if BatchSize-I<NRead[Sample]:
                    NRead[Sample]=BatchSize-I
#                print Sample,Status
         
                FileName=Files[Sample][Status[Sample]["fileI"]]
#                print "Opening File:",FileName
                f=h5py.File(FileName,"r")

                NEvents_File=f["features"].shape[0]

                StartI=Status[Sample]["eventI"]

#                print "NRead[Sample]", NRead[Sample]
#                print "NEvents_File" , NEvents_File 
#                print "StartI      " , StartI       

                if NRead[Sample]<NEvents_File-StartI:
                    NEnd=StartI+NRead[Sample]
                    Status[Sample]["eventI"]+=NRead[Sample]                                        
                else:
                    NEnd=NEvents_File
                    Status[Sample]["eventI"]=0
                    Status[Sample]["fileI"]+=1

                NTotal+=NEnd-StartI

#                print "NEnd",NEnd
#                print "NTotal",NTotal
                    

                TheShape=(BatchSize,)+ f["features"].shape[1:]

                if First:
                    Data_X=np.zeros(TheShape)
                    Data_Y=np.zeros((BatchSize) )
                    First=False
                
                Data_X[I:I+(NEnd-StartI)]=f["features"][StartI:NEnd]
                a=np.empty(NEnd-StartI); a.fill(ClassIndex[Sample])
#                print a.shape
#                print Data_Y.shape
#                print I
                Data_Y[I:I+(NEnd-StartI)]=a
                I+=NEnd-StartI
                f.close()

        Data_X,Data_Y=shuffle_in_unison_inplace(Data_X,Data_Y)
        print "t=",time()-start, 
                            
        yield (Data_X,Data_Y),ClassIndex


xx=DataGenerator(Samples,BatchSize=1024)

I=0
for (x,y),ClassIndex in xx:
    print x.shape,y.shape,ClassIndex
    if I>5:
        break
    I+=1
        
#(Data_X,Data_Y_),ClassIndex=LoadData(Samples,10000)



# def LArIATDataGenerator(InSamples,FileSearch="/data/LArIAT/h5_files",
#                         MaxFiles=-1, verbose=True, OneHot=True, ClassIndex=False,
#                         ClassIndexMap=False,batchsize=2048):
    
#     Files={}
#     NEvents={}
#     ClassIndex={}
    
#     NSamples=len(InSamples)

#     for I,Sample in enumerate(InSamples):
#         Files[Sample]=glob.glob(FileSearch+"/*"+Sample+"*.h5")
#         NEvents[Sample]=0
#         ClassIndex[Sample]=I

#     FileCount=0

#     Samples=[]
#     for Sample in InSamples:
#         for File in Files[Sample]:
            
#             FileCount+=1
#             ParticleName=Sample

#             Samples.append((File,"features",ParticleName))

#         if MaxFiles>0:
#             if FileCount>MaxFiles:
#                 break

#     return MultiClassGenerator(Samples,batchsize,
#                                verbose=verbose, 
#                                OneHot=OneHot,
#                                ClassIndex=ClassIndex, 
#                                Energy=False, 
#                                ClassIndexMap=ClassIndexMap)



