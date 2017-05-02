import sys,os,argparse

# Parse the Arguments
execfile("LArTPCDNN/ClassificationArguments.py")

# Process the ConfigFile
execfile(ConfigFile)

# Load the Data
from LArTPCDNN.LoadData import * 

TrainSampleList,TestSampleList=DivideFiles(FileSearch,[float(NSamples)/MaxEvents,float(NTestSamples)/MaxEvents],
                                           datasetnames=[u'features'],
                                           Particles=Particles)

# Figure out the output shape... This is not necessary. But the automatic mechanism is inefficient.
if ScanWindowSize>0:
#    shapes=[(BatchSize*multiplier, 2, 240, ScanWindowSize), (BatchSize*multiplier, NClasses)]
    shapes=[(BatchSize*multiplier, 240, ScanWindowSize),
            (BatchSize*multiplier, 240, ScanWindowSize),
            (BatchSize*multiplier, NClasses)]
    viewshape=(None, 240, ScanWindowSize)
else:
    shapes=[(BatchSize*multiplier, 240, 4096/DownSampleSize),
            (BatchSize*multiplier, 240, 4096/DownSampleSize),
            (BatchSize*multiplier, NClasses)]

    viewshape=(None, 240, 4096/DownSampleSize)

def MakeGenerator(SampleList,NSamples,
                  cachefile="LArIAT-LoadDataTest-Cache.h5",**kwargs):

    return DLMultiClassFilterGenerator(TrainSampleList, FilterEnergy(EnergyCut), max=NSamples,
                                       preprocessfunction=ProcessWireData(DownSampleSize,ScanWindowSize,Normalize),
                                       postprocessfunction=MergeInputs(),
                                       batchsize=BatchSize,
                                       shapes=shapes,
                                       n_threads=n_threads,
                                       multiplier=multiplier,
                                       cachefile=cachefile,
                                       **kwargs)

# Use DLGenerators to read data
Train_genC = MakeGenerator(TrainSampleList, NSamples,
                           cachefile="/tmp/LArTPCDNN-LArIAT-TrainEvent-Cache.h5")

Test_genC = MakeGenerator(TestSampleList, NTestSamples,
                          cachefile="/tmp/LArTPCDNN-LArIAT-TestEvent-Cache.h5")

print "Train Class Index Map:", Train_genC.ClassIndexMap
#print "Test Class Index Map:", Test_genC.ClassIndexMap

Cache=True

if Preload:
    print "Caching data in memory for faster processing after first epoch. Hope you have enough memory."
    Train_gen=Train_genC.PreloadGenerator()
    Test_gen=Test_genC.PreloadGenerator()
elif Cache:
    print "Caching data on disk for faster processing after first epoch. Hope you have enough disk space."
    Train_gen=Train_genC.DiskCacheGenerator(n_threads_cache)
    Test_gen=Test_genC.DiskCacheGenerator(n_threads_cache)
else:
    Train_gen=Train_genC.Generator()
    Test_gen=Test_genC.Generator()


# Build/Load the Model
from DLTools.ModelWrapper import ModelWrapper
from LArTPCDNN.Models import *

# You can automatically load the latest previous training of this model.
if TestDefaultParam("LoadPreviousModel") and not LoadModel:
    print "Looking for Previous Model to load."
    ModelName=Name
    if View1 and View2:
        ModelName+="_Merged"
    MyModel=ModelWrapper(Name=ModelName, LoadPrevious=True,OutputBase=OutputBase)

# You can load a previous model using "-L" option with the model directory.
if LoadModel:    
    print "Loading Model From:",LoadModel
    if LoadModel[-1]=="/": LoadModel=LoadModel[:-1]
    MyModel=ModelWrapper(Name=os.path.basename(LoadModel),InDir=os.path.dirname(LoadModel),
                         OutputBase=OutputBase)
    MyModel.Load(LoadModel)

if not MyModel.Model:
    FailedLoad=True
else:
    FailedLoad=False

# Or Build the model from scratch
if FailedLoad:
    import keras
    print "Building Model...",

    if View1:
        View1Model=FullyConnectedClassification(Name+"View1", viewshape, Width, Depth,
                                                BatchSize, NClasses,
                                                init=TestDefaultParam("WeightInitialization",'normal'),
                                                activation=TestDefaultParam("activation","relu"),
                                                Dropout=TestDefaultParam("DropoutLayers",0.5),
                                                BatchNormalization=TestDefaultParam("BatchNormLayers",False),
                                                NoClassificationLayer=View1 and View2,
                                                OutputBase=OutputBase)
        View1Model.Build()
        MyModel=View1Model

    if View2:
        View2Model=FullyConnectedClassification(Name+"View2", viewshape, Width, Depth,
                                                BatchSize, NClasses,
                                                init=TestDefaultParam("WeightInitialization",'normal'),
                                                activation=TestDefaultParam("activation","relu"),
                                                Dropout=TestDefaultParam("DropoutLayers",0.5),
                                                BatchNormalization=TestDefaultParam("BatchNormLayers",False),
                                                NoClassificationLayer=View1 and View2,
                                                OutputBase=OutputBase)
        View2Model.Build()
        MyModel=View2Model

    if View1 and View2:
        MyModel=MergerModel(Name+"_Merged",[View1Model,View2Model], NClasses, WeightInitialization,
                            OutputBase=OutputBase)

    # Configure the Optimizer, using optimizer configuration parameter.
    MyModel.Loss=loss
    # Build it
    MyModel.Build()
    print " Done."


print "Output Directory:",MyModel.OutDir
# Store the Configuration Dictionary
MyModel.MetaData["Configuration"]=Config
if "HyperParamSet" in dir():
    MyModel.MetaData["HyperParamSet"]=HyperParamSet

# Print out the Model Summary
MyModel.Model.summary()

# Compile The Model
print "Compiling Model."
MyModel.BuildOptimizer(optimizer,Config)
MyModel.Compile(Metrics=["accuracy"]) 

# Train
if Train or (RecoverMode and FailedLoad):
    print "Training."
    # Setup Callbacks
    # These are all optional.
    from DLTools.CallBacks import TimeStopping, GracefulExit
    from keras.callbacks import *
    callbacks=[ ]

    # Still testing this...

    if TestDefaultParam("UseGracefulExit",0):
        print "Adding GracefulExit Callback."
        callbacks.append( GracefulExit() )

    if TestDefaultParam("ModelCheckpoint",False):
        MyModel.MakeOutputDir()
        callbacks.append(ModelCheckpoint(MyModel.OutDir+"/Checkpoint.Weights.h5",
                                         monitor=TestDefaultParam("monitor","val_loss"), 
                                         save_best_only=TestDefaultParam("ModelCheckpoint_save_best_only"),
                                         save_weights_only=TestDefaultParam("ModelCheckpoint_save_weights_only"),
                                         mode=TestDefaultParam("ModelCheckpoint_mode","auto"),
                                         period=TestDefaultParam("ModelCheckpoint_period",1),
                                         verbose=0))

    if TestDefaultParam("EarlyStopping"):
        callbacks.append(keras.callbacks.EarlyStopping(monitor=TestDefaultParam("monitor","val_loss"), 
                                                       min_delta=TestDefaultParam("EarlyStopping_min_delta",0.01),
                                                       patience=TestDefaultParam("EarlyStopping_patience"),
                                                       mode=TestDefaultParam("EarlyStopping_mode",'auto'),
                                                       verbose=0))


    if TestDefaultParam("RunningTime"):
        print "Setting Runningtime to",RunningTime,"."
        TSCB=TimeStopping(TestDefaultParam("RunningTime",3600*6),verbose=False)
        callbacks.append(TSCB)
    

    # Don't fill the log files with progress bar.
    if sys.flags.interactive:
        verbose=1
    else:
        verbose=1 # Set to 2

    print "Evaluating score on test sample..."
    score = MyModel.Model.evaluate_generator(Test_gen, steps=NTestSamples/BatchSize)
    
    print "Initial Score:", score
    MyModel.MetaData["InitialScore"]=score
        
    MyModel.History = MyModel.Model.fit_generator(Train_gen,
                                                  steps_per_epoch=(NSamples/BatchSize),
                                                  epochs=Epochs,
                                                  verbose=verbose, 
                                                  validation_data=Test_gen,
                                                  validation_steps=NTestSamples/BatchSize,
                                                  callbacks=callbacks)

    score = MyModel.Model.evaluate_generator(Test_gen, steps=NTestSamples/BatchSize)


    print "Evaluating score on test sample..."
    print "Final Score:", score
    MyModel.MetaData["FinalScore"]=score

    if TestDefaultParam("RunningTime"):
        MyModel.MetaData["EpochTime"]=TSCB.history

    # Store the parameters used for scanning for easier tables later:
    for k in Params:
        MyModel.MetaData[k]=Config[k]

    # Save Model
    MyModel.Save()
else:
    print "Skipping Training."
    
# Analysis
if Analyze:
    Test_genC = MakeGenerator(TestSampleList, NTestSamples,
                              cachefile=Test_genC.cachefilename) #"/tmp/LArTPCDNN-LArIAT-TestEvent-Cache.h5")

    Test_genC.PreloadData(n_threads_cache)
    [Test_X_View1, Test_X_View2], Test_Y = MergeInputs()(tuple(Test_genC.D))

    from DLAnalysis.Classification import MultiClassificationAnalysis
    result,NewMetaData=MultiClassificationAnalysis(MyModel,[Test_X_View1,Test_X_View2],
                                                   Test_Y,BatchSize,PDFFileName="ROC",
                                                   IndexMap=Test_genC.ClassIndexMap)

    MyModel.MetaData.update(NewMetaData)
    
    # Save again, in case Analysis put anything into the Model MetaData
    if not sys.flags.interactive:
        MyModel.Save()
    else:
        print "Warning: Interactive Mode. Use MyModel.Save() to save Analysis Results."
        
# Make sure all of the Generators processes and threads are dead.
# Not necessary... but ensures a graceful exit.
# if not sys.flags.interactive:
#     for g in GeneratorClasses:
#         try:
#             g.StopFiller()
#             g.StopWorkers()
#         except:
#             pass
