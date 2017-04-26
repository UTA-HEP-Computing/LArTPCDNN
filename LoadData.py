import h5py
import glob,os,sys,time
import numpy as np

from DLTools.ThreadedGenerator import DLMultiClassFilterGenerator

#[u'Eng', u'Track_length', u'enu_truth', u'features', u'lep_mom_truth', u'mode_truth', u'pdg']

def shuffle_in_unison_inplace(a, b, c=False):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    if type(c) != bool:
        return a[p], b[p], c[p]
    return a[p], b[p]

def DownSample(y,factor,batchsize,sumabs=False):
    Nx=batchsize
    Ny=y.shape[1]
    Nz=y.shape[2]
    Nw=y.shape[3]
    
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

def GetXWindow(Data,i,BoxSizeX):
    return Data[:,:,i:i+BoxSizeX]

def ScanWindow(inData,BoxSizeX=256,Nx=240,Ny=4096):
    NyNew=Ny
    #Scan the Window

    first=True
    out=False
    
    for II in xrange(inData.shape[0]):
        Data=inData[II]
        b=np.array([0]*(NyNew-BoxSizeX))

        for i in xrange(0,NyNew-BoxSizeX):
            b[i]=GetXWindow(Data,i,BoxSizeX).clip(0,99999999999).sum()

        #Find the Window with max Energy/Charge
        BoxStart=b.argmax()
        MaxSum=b[BoxStart]

        #Store the window
        if first:
            first=False
            Box=Data[:,:,BoxStart:BoxStart+BoxSizeX]
            out=np.zeros((inData.shape[0],)+Box.shape)
            out[II]=Box
        else:
            out[II]=Data[:,:,BoxStart:BoxStart+BoxSizeX]

    return out,BoxStart,MaxSum
     
def FilterEnergy(MinEnergy):
    def filterfunction(batchdict):
        r= np.where(np.array(batchdict['Eng']) > MinEnergy)
        return r[0]

    return filterfunction

def ProcessWireData(DownSampleFactor,ScanWindowSize,Norm=True):
    def processfunction(D):
        X=D[0]
        BatchSize=X.shape[0]
        if DownSampleFactor > 1:
            X,Ny= DownSample(X,DownSampleFactor,BatchSize)
        if ScanWindowSize>0:
            X,i,j=ScanWindow(X,ScanWindowSize,240,Ny)

        if Norm:
            X = np.tanh(np.sign(X) * np.log(np.abs(X) + 1.0) / 2.0)
        return [X]+D[1:]
    return processfunction
    

def LArIATDataGenerator(FileSearch="/data/LArIAT/*.h5",DownSampleSize=4, ScanWindowSize=256,EnergyCut=0.61,
                        datasetnames=[u'features'], Norm=False, MaxFiles=-1, **kwargs):

    print "Searching in :",FileSearch
    Files = glob.glob(FileSearch)

    print "Found",len(Files),"files."
    
    if MaxFiles!=-1:
        random.shuffle(Files)
        Files=Files[:MaxFiles]
        
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
    
    GC= DLMultiClassFilterGenerator(Samples, FilterEnergy(EnergyCut), 
                                    preprocessfunction=ProcessWireData(DownSampleSize,ScanWindowSize,Norm),
                                    **kwargs)
    GC.datasets=datasetnames
    return GC

if __name__ == '__main__':
    import sys
    FileSearch="/data/LArIAT/h5_files/*.h5"

    try:
        n_threads=int(sys.argv[1])
    except:
        n_threads=20

    try:
        n_threads2=int(sys.argv[2])
    except:
        n_threads2=n_threads

    closefiles=False
    if n_threads>=61:
        closefiles=True
        
    print "Building Generator"
    sys.stdout.flush()
    m=1
    DownSampleSize=8
    ScanWindowSize=256
    Normalize=True
    closefiles=False
    Train_gen=LArIATDataGenerator(FileSearch=FileSearch,
                                  cachefile="LArIAT-LoadDataTest-Cache.h5",
                                  max=128*10000, 
                                  batchsize=128,
                                  DownSampleSize=DownSampleSize,
                                  ScanWindowSize=ScanWindowSize,
                                  Norm=Normalize,
                                  #shapes=[(128*m, 2, 240, 4096/DownSampleSize), (128*m, 16)],
                                  shapes=[(128*m, 2, 240, ScanWindowSize), (128*m, 16)],
                                  n_threads=n_threads,
                                  SharedDataQueueSize=1,
                                  multiplier=m,
                                  closefiles=closefiles,
                                  verbose=False,
                                  timing=False,
                                  sleep=1,
                                  Wrap=False)

    print "Generator Ready"
    print "ClassIndex:", Train_gen.ClassIndexMap
    print "Object Shape:",Train_gen.shapes

    N=1
    NN=n_threads
    count=0
    old=start=time.time()
    for tries in xrange(2):
        print "*********************Try:",tries
        #for D in Train_gen.Generator():
        for D in Train_gen.DiskCacheGenerator(n_threads2):
            NN-=0
            if NN<0:
                break
            start1=time.time()
            Delta=(start1-start)
            Delta2=(start1-old)
            old=start1
            print count,":",Delta, ":",Delta/float(N), Delta2
            sys.stdout.flush()
            N+=1
            for d in D:
                print d.shape
                #print d[np.where(d!=0.)]
                NN=d.shape[0]
                #print d[0]
                pass
            count+=NN
