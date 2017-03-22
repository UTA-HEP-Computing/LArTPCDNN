import h5py
import glob,os,sys,time
import numpy as np

from DLTools.ThreadedGeneratorTest import DLMultiClassGenerator
from DLTools.ThreadedGeneratorTest import DLh5FileGenerator

#[u'Eng', u'Track_length', u'enu_truth', u'features', u'lep_mom_truth', u'mode_truth', u'pdg']

Samples = [ "electron", "pion_0"]

def shuffle_in_unison_inplace(a, b, c=False):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    if type(c) != bool:
        return a[p], b[p], c[p]
    return a[p], b[p]

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

def DataGenerator(Samples,BatchSize=1024,DownSampleFactor=2,path="/data/LArIAT/h5_files"):
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

                InShape=f["features"].shape
                
                Nx = BatchSize
                Ny = InShape[1]
                Nz = InShape[2]
                Nw = InShape[3]
                
                DataIn=f["features"][StartI:NEnd]
                DataInDS,NwNew=DownSample(DataIn,DownSampleFactor,Nx,Ny,Nz,Nw)
                
                TheShape=(BatchSize,)+ DataInDS.shape[1:]

                if First:
                    Data_X=np.zeros(TheShape)
                    Data_Y=np.zeros((BatchSize) )
                    First=False
                
                Data_X[I:I+(NEnd-StartI)]=DataInDS
                
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


#xx=DataGenerator(Samples,BatchSize=1024)
#
#I=0
#for (x,y),ClassIndex in xx:
#    print x.shape,y.shape,ClassIndex
#    if I>5:
#        break
#    I+=1


def LArIATDataGenerator(datasetnames=[u'features'],batchsize=2048,FileSearch="/data/LArIAT/*.h5",MaxFiles=-1,
                        verbose=False, OneHot=True, ClassIndex=False, ClassIndexMap=False,n_threads=4,
                        multiplier=1,timing=False,**kwargs):
    print "Searching in :",FileSearch
    Files = glob.glob(FileSearch)

    #if MaxFiles!=-1:
    #    random.shuffle(Files)
    Samples=[]

    FileCount=0

    for F in Files:
        FileCount+=1
        basename=os.path.basename(F)
        ParticleName=basename.split("_")[0]

        Samples.append((F,datasetnames,ParticleName))
        if MaxFiles>0:
            if FileCount>MaxFiles:
                break

    
    GC= DLMultiClassGenerator(Samples,batchsize,
                              verbose=verbose, 
                              #OneHot=OneHot,
                              ClassIndex=ClassIndex,
                              n_threads=n_threads,
                              multiplier=multiplier,
                              timing=timing, **kwargs)

            
    if ClassIndexMap:
        return [GC,GC.ClassIndexMap]
    else:
        return GC

if __name__ == '__main__':
    import sys
    FileSearch="/data/LArIAT/h5_files/*.h5"

    try:
        n_threads=int(sys.argv[1])
    except:
        n_threads=4

    [Train_gen, CM]=LArIATDataGenerator(FileSearch=FileSearch,batchsize=128,n_threads=n_threads,ClassIndexMap=True)#,verbose=True)

    print CM

    N=1
    count=0
    start=time.time()
    for tries in xrange(2):
        print "*********************Try:",tries
        for D in Train_gen.Generator():
            Delta=(time.time()-start)
            print count,":",Delta, ":",Delta/float(N)
            sys.stdout.flush()
            N+=1
            for d in D:
                print d.shape
                print d[np.where(d!=0.)]
                NN=d.shape[0]
                #print d[0]
                pass
            count+=NN
