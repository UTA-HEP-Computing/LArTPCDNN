import h5py
import glob, os, sys, time
import numpy as np

from DLTools.ThreadedGenerator import DLMultiClassGenerator, DLMultiClassFilterGenerator


def main():
    datapath = "/data/datasets/LarTPC/apr_9/"

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


def LarTPCDataGenerator(files="/data", batchsize=16, datasets=['images3D/C', 'images3D/V'], Norm=True,
                        bins=(240, 240, 256), **kwargs):
    Samples = []

    for F in files:
        basename = os.path.basename(F)
        ParticleName = basename.split("_")[0]
        Samples.append((F, datasets, ParticleName))

    # Samples = [ (Directory+"muon_158.2d.h5", datasets, "data")]

    def MakeImage(bins, Norm=True):
        def f(D):
            for i in xrange(D[0].shape[0]):
                if Norm:
                    w = np.tanh(np.sign(D[1][i]) * np.log(np.abs(D[1][i]) + 1.0) / 2.0)
                else:
                    w = D[1][i]
                R, b = np.histogramdd(D[0][i], bins=list(bins), weights=w)
            return [R] + D[2:]

        return f

    GC = DLMultiClassGenerator(Samples, batchsize=batchsize,
                               preprocessfunction=MakeImage(bins, False),
                               OneHot=True,
                               shapes=[(batchsize,) + bins, (batchsize, 2)],
                               **kwargs)
    return GC


#
# def LarTPCDataGeneratorOld(Directory="/data/", batchsize=16, datasets=[u'3DImages'],**kwargs):
#
#    Samples = [ (Directory+"3d/", datasets, "signal"    ),
#                (Directory+"/dnn_NEXT100_Bi214_bg_v2x2x2_r200x200x200.Tensor.h5", datasets, "background"    )]
#
#    def filterfunction(batchdict):
#        r= np.array(range(batchdict["3DImages"].shape[0]))
#        return r[0]
#
#    
#    GC= DLMultiClassFilterGenerator(Samples, batchsize=batchsize, FilterFunc=False,
#                                    OneHot=True, shapes = [(batchsize, 200,200,200), (batchsize, 2)],  **kwargs)
#    return GC




# Test=1

# if __name__ == '__main__' and Test==0:
#    import sys
#    Directory="/data/datasets/LarTPC/apr_9/3d/"
#
#    try:
#        n_threads=int(sys.argv[1])
#    except:
#        n_threads=6
#
#    try:
#        n_threads2=int(sys.argv[2])
#    except:
#        n_threads2=n_threads
#
#    Train_gen=LarTPCDataGeneratorOld(Directory,n_threads=n_threads,max=100000, verbose=False)
#    
#    print "Generator Ready"
#    print "ClassIndex:", Train_gen.ClassIndexMap
#    print "Object Shape:",Train_gen.shapes
#    sys.stdout.flush()
#    
#    N=1
#    NN=n_threads
#    count=0
#    old=start=time.time()
#    for tries in xrange(1):
#        print "*********************Try:",tries
#        #for D in Train_gen.Generator():
#        for D in Train_gen.Generator():
#            NN-=0
#            if NN<0:
#                break
#            start1=time.time()
#            Delta=(start1-start)
#            Delta2=(start1-old)
#            old=start1
#            print count,":",Delta, ":",Delta/float(N), Delta2
#            sys.stdout.flush()
#            N+=1
#            for d in D:
#                print d.shape
#                #print d[np.where(d!=0.)]
#                NN=d.shape[0]
#                #print d[0]
#                pass
#            count+=NN


if __name__ == '__main__':
    main()
