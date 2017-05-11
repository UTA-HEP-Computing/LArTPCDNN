import h5py
import glob, os, sys, time
import numpy as np

from DLTools.ThreadedGenerator import DLMultiClassGenerator, DLMultiClassFilterGenerator

def combined2D3DGenerator():
    #datapath = "/data/datasets/LarTPC/apr_9/"
    datapath = "/data/cloud/project/data/apr_9/"
    # Pull in datafiles
    filelist2d = glob.glob(datapath + "2d/*")
    filelist3d = glob.glob(datapath + "3d/*")
    filelist2d.sort()
    filelist3d.sort()
    assert len(filelist2d) == len(filelist3d), "Number of 2D and 3D files mismatch!"

    try:
        n_threads = int(sys.argv[1])
    except:
        n_threads = 6

    try:
        n_threads2 = int(sys.argv[2])
    except:
        n_threads2 = n_threads


    Train_gen3D = LarTPCDataGenerator(filelist3d, n_threads=n_threads, max=100000, 
                                    bins=(240, 240, 256), verbose=False)

    DownSampleSize=8
    ScanWindowSize=256
    Normalize=True
    closefiles=False
    m = 1
    Train_gen2D =LArIATDataGenerator(FileSearch=datapath + "2d/*",
                                  max=128*10000, 
                                  batchsize=128,
                                  DownSampleSize=DownSampleSize,
                                  ScanWindowSize=ScanWindowSize,
                                  Norm=Normalize,
                                  #shapes=[(128*m, 2, 240, 4096/DownSampleSize), (128*m, 16)],
                                  #shapes=[(128*m, 240, ScanWindowSize), (128, 240, 256)],
                                  #shapes=[(128*m, 2, 240, ScanWindowSize), (128*m, 16)], 
                                  #shapes=[(128*m, 240, ScanWindowSize)],
                                  n_threads=n_threads,
                                  SharedDataQueueSize=1,
                                  multiplier=m,
                                  closefiles=closefiles,
                                  verbose=False,
                                  timing=False,
                                  sleep=1,
                                  Wrap=False)


    def MergerGenerator(T2D, T3D):
        while True:
            s2d = T2D.Generator().next()
            s3d = T3D.Generator().next()
            if s2d and s3d:
                yield [s2d[0], s2d[1], s3d[0]]
            else:
                break

    return MergerGenerator(Train_gen2D, Train_gen3D)



def main():
    #datapath = "/data/datasets/LarTPC/apr_9/"
    datapath = "/data/cloud/project/data/apr_9/"
    # Pull in datafiles
    filelist2d = glob.glob(datapath + "2d/*")
    filelist3d = glob.glob(datapath + "3d/*")
    filelist2d.sort()
    filelist3d.sort()
    assert len(filelist2d) == len(filelist3d), "Number of 2D and 3D files mismatch!"

    try:
        n_threads = int(sys.argv[1])
    except:
        n_threads = 6

    try:
        n_threads2 = int(sys.argv[2])
    except:
        n_threads2 = n_threads


    Train_gen = LarTPCDataGenerator(filelist3d, n_threads=n_threads, max=100000, 
                                    bins=(240, 240, 256), verbose=False)

    DownSampleSize=8
    ScanWindowSize=256
    Normalize=True
    closefiles=False
    m = 1

    print "Generator Ready"
    print "ClassIndex:", Train_gen.ClassIndexMap
    print "Object Shape:", Train_gen.shapes
    sys.stdout.flush()

    N = 1
    NN = n_threads
    count = 0
    old = start = time.time()
    for tries in xrange(1):
        print "*********************Try:", tries
        # for D in Train_gen.Generator():
        # GENERATOR CALLED HERE, FEED THIS TO OUTPUT FOR 3D
        for D in Train_gen.Generator():
            NN -= 0
            if NN < 0:
                break
            start1 = time.time()
            Delta = (start1 - start)
            Delta2 = (start1 - old)
            old = start1
            print count, ":", Delta, ":", Delta / float(N), Delta2
            sys.stdout.flush()
            N += 1
            for d in D:
                print d.shape
                print d[np.where(d != 0.)]
                NN = d.shape[0]
                # print d[0]
                pass
            count += NN


def LArIATDataGenerator(FileSearch="/data/LArIAT/*.h5",DownSampleSize=4, ScanWindowSize=256,EnergyCut=0.61,
                        datasetnames=[u'features'], Norm=False, MaxFiles=-1, **kwargs):

    print "Searching in :",FileSearch
    Files = glob.glob(FileSearch)

    print "Found",len(Files),"files."
    Files.sort()
    
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
    return GC






def LarTPCDataGenerator(files="/data", is2D = False, batchsize=16, datasets=['images3D/C', 'images3D/V'], Norm=True,
                        bins=None, EnergyCut=0.61, DownSampleSize = 2, ScanWindowSize = 256, **kwargs):
    Samples = []

    for F in files:
        basename = os.path.basename(F)
        ParticleName = basename.split("_")[0]
        Samples.append((F, datasets, ParticleName))

    # Samples = [ (Directory+"muon_158.2d.h5", datasets, "data")]

    def MakeImage(bins, Norm=True):
        if bins != None:
            def f(D):
                for i in xrange(D[0].shape[0]):
                    if Norm:
                        w = np.tanh(np.sign(D[1][i]) * np.log(np.abs(D[1][i]) + 1.0) / 2.0)
                    else:
                        w = D[1][i]
                    R, b = np.histogramdd(D[0][i], bins=list(bins), weights=w)
                return [R] + D[2:]
            return f
        else: 
            return False
        
    if bins == None:
        bins = (0,)

    if is2D:
        GC= DLMultiClassFilterGenerator(Samples, FilterEnergy(EnergyCut), 
                                preprocessfunction=ProcessWireData(DownSampleSize,ScanWindowSize,Norm),
                                **kwargs)
    
    else:
        GC = DLMultiClassGenerator(Samples, batchsize=batchsize,
                                   preprocessfunction=MakeImage(bins, False),
                                   OneHot=True,
                                   shapes=[(batchsize,) + bins, (batchsize, 2)],
                                   **kwargs)

    return GC


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
            #X,i,j=ScanWindow(X,ScanWindowSize,240,Ny)
            X=crop_batch(X,ScanWindowSize)

        if Norm:
            X = np.tanh(np.sign(X) * np.log(np.abs(X) + 1.0) / 2.0)
        return [X[:,0,:,:],X[:,1,:,:]] +D[1:]
    return processfunction

# From Peter Sadowski
def crop_example(X, interval, augment=None):
    ''' 
    Crop X by finding time interval with maximal energy. 
    X  = tensor of shape (num_channel, x, y) = (2 channels, 240 wires, time steps)
    interval = length of desired time step window
    augment  = If integer, randomly translate the time window up to this many steps.
    '''
    assert len(X.shape) == 3, "Example is expected to be three-dimensional."
    energy = np.sum(X, axis=(0,1))
    assert energy.ndim == 1
    cumsum = np.cumsum(energy, dtype='float64')
    assert not np.any(np.isnan(cumsum))
    assert np.all(np.isfinite(cumsum))
    intsum = cumsum[interval:] - cumsum[:-interval]
    maxstart = np.argmax(intsum) # NOTE: maxend=interval+np.argmax(intsum)
    
    if augment:
        rsteps = np.random.random_integers(-augment, augment)
        if rsteps < 0:
            maxstart = max(0, maxstart + rsteps)
        else:
            maxstart = min(len(energy)-interval, maxstart + rsteps)

    return X[:, :, maxstart:maxstart+interval]

def crop_batch(X, interval, augment=None):
    new_X = np.zeros(shape=(X.shape[0],X.shape[1],X.shape[2],interval), dtype='float32')
    for i in range(X.shape[0]):
        new_X[i,:,:,:] = crop_example(X[i,:,:,:], interval, augment)
    return new_X


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




#if __name__ == '__main__':
#    main()
