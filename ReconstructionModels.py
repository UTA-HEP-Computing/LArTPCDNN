from DLTools.ModelWrapper import *

from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import  BatchNormalization,Dropout,Flatten, Input
from keras.models import model_from_json

class Model2DViewsTo3D(ModelWrapper):
    def __init__(self, Name, View1Shape, View2Shape, width=0, depth=0, BatchSize=2048, N_Classes=0,
                 init=0, BatchNormalization=False, Dropout=False, **kwargs):
        super(Model2DViewsTo3D, self).__init__(Name,**kwargs)
        
        
        self.width=width
        self.depth=depth
        self.init=init
        
        self.Dropout=Dropout
        self.BatchSize=BatchSize
        self.BatchNormalization=BatchNormalization
        
        self.input1_shape = View1Shape
        self.input2_shape = View2Shape
        self.N_Classes = N_Classes
        
        self.MetaData.update({ "width":self.width,
                               "depth":self.depth,
                               "Dropout":self.Dropout,
                               "BatchNormalization":BatchNormalization,
                               "input1_shape":self.input1_shape,
                               "input2_shape":self.input2_shape,
                               "N_classes":self.N_Classes,
                               "init":self.init})

    def Build(self):
        input1=Input(self.input1_shape)
        input2=Input(self.input2_shape)
        input1 = Flatten(input_shape=self.input1_shape)(input1)
        input2 = Flatten(input_shape=self.input2_shape)(input2)
        modelT = concatenate([input1, input2])
        
        #model.add(Dense(self.width,init=self.init))
        modelT = (Activation('relu')(modelT))

        for i in xrange(0,self.depth):
            if self.BatchNormalization:
                modelT=BatchNormalization()(modelT)

            modelT=Dense(self.width,kernel_initializer=self.init)(modelT)
            modelT=Activation(self.Activation)(modelT)

            if self.Dropout:
                modelT=Dropout(self.Dropout)(modelT)

        modelT=Dense(self.N_Classes, activation='softmax',kernel_initializer=self.init)(modelT)
        
        self.Model=Model(input,modelT)