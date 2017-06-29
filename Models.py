from DLTools.ModelWrapper import *

#from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import  BatchNormalization,Dropout,Flatten, Input
from keras.models import model_from_json
from keras.layers import Input, Dense, Flatten, Reshape, merge, Activation, \
        Convolution2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Dropout

import numpy as np

# class FullyConnectedClassification(ModelWrapper):
#     def __init__(self, Name, input_shape, width=0, depth=0, BatchSize=2048,
#                  N_classes=100, init=0, BatchNormalization=False, Dropout=False,
#                  NoClassificationLayer=False,
#                  activation='relu',**kwargs):

#         super(FullyConnectedClassification, self).__init__(Name,**kwargs)

#         self.width=width
#         self.depth=depth
#         self.input_shape=input_shape
#         self.N_classes=N_classes
#         self.init=init

#         self.Dropout=Dropout

#         self.BatchSize=BatchSize
#         self.BatchNormalization=BatchNormalization
#         self.Activation=activation
#         self.NoClassificationLayer=NoClassificationLayer
        
#         self.MetaData.update({ "width":self.width,
#                                "depth":self.depth,
#                                "Dropout":self.Dropout,
#                                "BatchNormalization":BatchNormalization,
#                                "input_shape":self.input_shape,
#                                "N_classes":self.N_classes,
#                                "init":self.init})
#     def Build(self):
#         input=Input(self.input_shape[1:])
#         modelT = Flatten(input_shape=self.input_shape)(input)

# #        model.add(Dense(self.width,init=self.init))
#         modelT = (Activation('relu')(modelT))

#         for i in xrange(0,self.depth):
#             if self.BatchNormalization:
#                 modelT=BatchNormalization()(modelT)

#             modelT=Dense(self.width,kernel_initializer=self.init)(modelT)
#             modelT=Activation(self.Activation)(modelT)

#             if self.Dropout:
#                 modelT=Dropout(self.Dropout)(modelT)

#         if not self.NoClassificationLayer:
#             modelT=Dense(self.N_classes, activation='softmax',kernel_initializer=self.init)(modelT)

#         self.inputT=input
#         self.modelT=modelT
        
#         self.Model=Model(input,modelT)

# class MergerModel(ModelWrapper):
#     def __init__(self, Name, Models, N_Classes, init, **kwargs):
#         super(MergerModel, self).__init__(Name,**kwargs)
#         self.Models=Models
#         self.N_Classes=N_Classes
#         self.init=init
        
#     def Build(self):

#         MModels=[]
#         MInputs=[]
#         for m in self.Models:
#             MModels.append(m.modelT)
#             MInputs.append(m.inputT)
#         if len(self.Models)>0:
#             print "Merged Models"
#             modelT=concatenate(MModels)#(modelT)
            
#         modelT=Dense(self.N_Classes, activation='softmax',kernel_initializer=self.init)(modelT)
        

#         self.modelT=modelT
        
#         self.Model=Model(MInputs,modelT)

def InceptionModule(L, C1_filters, C3r_filters, C3_filters, 
                    C5r_filters, C5_filters, CPr_filters,
                    init, activation, dim_ordering='th'):
    """ Function that adds Inception Module to network. """

    # Assign concat axis based on dim_ordering
    if dim_ordering == 'th':
        concat_axis = 1
    elif dim_ordering == 'tf':
        concat_axis = 3
    else:
        raise ValueError("InceptionModule dim_ordering must be 'th' or 'tf'.")
   
    # 1x1 convolutions
    C1 = Convolution2D(C1_filters, 1, 1, 
                       init=init, activation=activation, border_mode='same', 
                       subsample=(1,1), dim_ordering=dim_ordering, bias=True, 
                       W_regularizer=None, b_regularizer=None)(L)

    # 3x3 convolutions
    C3r = Convolution2D(C3r_filters, 1, 1, 
                       init=init, activation=activation, border_mode='same', 
                       subsample=(1,1), dim_ordering=dim_ordering, bias=True, 
                       W_regularizer=None, b_regularizer=None)(L)
    # 3x3 convolutions
    C3 = Convolution2D(C3_filters, 3, 3, 
                       init=init, activation=activation, border_mode='same', 
                       subsample=(1,1), dim_ordering=dim_ordering, bias=True, 
                       W_regularizer=None, b_regularizer=None)(C3r)

    # 5x5 convolutions
    C5r = Convolution2D(C5r_filters, 1, 1, 
                       init=init, activation=activation, border_mode='same', 
                       subsample=(1,1), dim_ordering=dim_ordering, bias=True, 
                       W_regularizer=None, b_regularizer=None)(L)
    C5 = Convolution2D(C5_filters, 5, 5, 
                       init=init, activation=activation, border_mode='same', 
                       subsample=(1,1), dim_ordering=dim_ordering, bias=True, 
                       W_regularizer=None, b_regularizer=None)(C5r)

    # 3x3 max pooling
    CP = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(L)
    CPr = Convolution2D(CPr_filters, 1, 1, 
                       init=init, activation=activation, border_mode='same', 
                       subsample=(1,1), dim_ordering=dim_ordering, bias=True, 
                       W_regularizer=None, b_regularizer=None)(CP)

    # Full inception module
    module = merge([C1, C3, C5, CPr], mode='concat', concat_axis=concat_axis)

    return module

def AuxiliaryClassifier(L, init, activation, output_shape, output_act, output_name):
    """ Function that adds auxiliary classifier to the network. """
    AP = AveragePooling2D(pool_size=(5,5), strides=(3,3), border_mode='valid')(L)
    C = Convolution2D(128, 1, 1,
                      init=init, activation=activation, border_mode='valid', 
                      subsample=(1,1), dim_ordering='th', bias=True, 
                      W_regularizer=None, b_regularizer=None)(AP)
    F = Flatten()(C)
    LN1 = Dense(256, activation=activation, init=init)(F)
    DO = Dropout(0.70)(LN1)
    #LN2 = Dense(num_classes, activation=activation, init='zero', name=output_name)(DO)
    LN2 = Dense(output_shape, activation=output_act, init='zero', name=output_name)(DO)
    return LN2
    ####^^^return DO

                
class SiameseInceptionClassification(ModelWrapper):
    def __init__(self, Name, input_shape, width=0, depth=0, BatchSize=2048,
                 N_classes=100, kernel_initializer=0, BatchNormalization=False, Dropout=False,
                 NoClassificationLayer=False,
                 activation='relu',nb_filter=np.array([32]),nb_row=np.array([4]),\
                 nb_column=np.array([4]),subsample=(1,1),output_shape=10,output_act='linear',
                 dim_ordering='th',
                 init="he_normal",
                 **kwargs):
####^^^^^change output_shape=2 --> output_shape = (128, 10)

        super(SiameseInceptionClassification, self).__init__(Name,**kwargs) ## , Loss, Optimizer

        self.dim_ordering=dim_ordering
        self.width=width
        self.depth=depth
        self.input_shape=input_shape
        self.N_classes=N_classes
        self.init=init
        self.kernel_initializer = kernel_initializer
        self.Dropout=Dropout

        self.BatchSize=BatchSize
        self.BatchNormalization=BatchNormalization
        self.Activation=activation
        self.NoClassificationLayer=NoClassificationLayer
        
        #####Petre's hp############
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_column = nb_column
        self.subsample = (1,1)
        ####******##########
        self.output_shape = output_shape
        self.output_act = output_act
        ############################
        
        self.MetaData.update({ "width":self.width,
                               "depth":self.depth,
                               "Dropout":self.Dropout,
                               "BatchNormalization":BatchNormalization,
                               "self.input_shape":self.input_shape,
                               "N_classes":self.N_classes,
                               #"init":self.init,
                               "kernel_initializer":self.kernel_initializer,
                               "nb_filter":self.nb_filter,
                               "nb_row": self.nb_row,
                               "nb_colomn":self.nb_column,
                               "output_act":self.output_act,
                               "output_shape":self.output_shape})
    def Build(self):
        # Input
        # Do not include batch size in input shape
        input_0 = Input(shape=self.input_shape, dtype='float32', name='input_0')
        # x0 = input_0
        # x0 = Convolution2D(self.nb_filter, (self.nb_row, self.nb_column),
        #                    kernel_initializer = self.kernel_initializer, activation=self.Activation, border_mode='same',
        #                    subsample=self.subsample, dim_ordering=self.dim_ordering, bias=True,
        #                    kernel_regularizer=None, bias_regularizer=None)(x0)


        # input_1 = Input(shape=self.input_shape, dtype='float32', name='input_1')

        # x1 = input_1
        # x1 = Convolution2D(self.nb_filter, (self.nb_row, self.nb_column),
        #                    kernel_initializer = self.kernel_initializer, activation=self.Activation, border_mode='same',
        #                    subsample=subsample, dim_ordering=self.dim_ordering, bias=True,
        #                    kernel_regularizer=None, bias_regularizer=None)(x1)

        # x = merge([x0, x1], mode='concat', concat_axis=1)

        # x = Convolution2D(self.nb_filter, 1, 1,
        #                   kernel_initializer = self.kernel_initializer, activation=self.Activation, border_mode='same',
        #                   subsample=(1, 1), dim_ordering=self.dim_ordering, bias=True,
        #                   kernel_regularizer=None, bias_regularizer=None)(x)
        # x = Convolution2D(64, 7, 7,
        #                   kernel_initializer = self.kernel_initializer, activation=self.Activation, border_mode='same',
        #                   subsample=(2,2), dim_ordering=self.dim_ordering, bias=True,
        #                   kernel_regularizer=None, bias_regularizer=None)(x)
        # x = ZeroPadding2D(padding=(1,1), dim_ordering=self.dim_ordering)(x)
        # x = MaxPooling2D(pool_size=(3,3), strides=(2,2),
        #                  border_mode='valid', dim_ordering=self.dim_ordering)(x)
        # x = Convolution2D(64, 1, 1,
        #                   kernel_initializer = self.kernel_initializer, activation=self.Activation, border_mode='same',
        #                   subsample=(1,1), dim_ordering=self.dim_ordering, bias=True,
        #                   kernel_regularizer=None, bias_regularizer=None)(x)
        # x = Convolution2D(192, 3, 3,
        #                   kernel_initializer = self.kernel_initializer, activation=self.Activation, border_mode='same',
        #                   subsample=(1,1), dim_ordering=self.dim_ordering, bias=True,
        #                   kernel_regularizer=None, bias_regularizer=None)(x)
        # x = ZeroPadding2D(padding=(1,1), dim_ordering=self.dim_ordering)(x)
        # x = MaxPooling2D(pool_size=(3,3), strides=(2,2),
        #                  border_mode='valid', dim_ordering=self.dim_ordering)(x)

        # x = InceptionModule(x, 64, 96, 128, 16, 32, 32, self.init, self.Activation)
        # x = InceptionModule(x, 128, 128, 192, 32, 96, 64, self.init, self.Activation)

        # x = MaxPooling2D(pool_size=(3,3), strides=(2,2),
        #                  border_mode='same', dim_ordering=self.dim_ordering)(x)

        # x = InceptionModule(x, 192, 96, 208, 16, 48, 64, self.init, self.Activation)
        
        # aux_output1 = AuxiliaryClassifier(x, kernel_initializer = self.kernel_initializer, activation=self.Activation,\
        #                                   output_shape=self.output_shape, output_act=self.output_act, output_name='aux_output1')
        # x = InceptionModule(x, 160, 112, 224, 24, 64, 64, self.init, self.Activation)
        # x = InceptionModule(x, 128, 128, 256, 24, 64, 64, self.init, self.Activation)
        # x = InceptionModule(x, 112, 144, 288, 32, 64, 64, self.init, self.Activation)
        
        # aux_output2 = AuxiliaryClassifier(x, kernel_initializer = self.kernel_initializer, activation=self.Activation, \
        #                                   output_shape=self.output_shape, output_act=self.output_act, output_name='aux_output2')
        # x = InceptionModule(x, 256, 160, 320, 32, 128, 128, self.init, self.Activation)
        # x = MaxPooling2D(pool_size=(3,3), strides=(2,2),
        #                  border_mode='same', dim_ordering=self.dim_ordering)(x)
        # x = InceptionModule(x, 256, 160, 320, 32, 128, 128, self.init, self.Activation)
        # x = InceptionModule(x, 384, 192, 384, 48, 128, 128, self.init, self.Activation)
        # x = AveragePooling2D(pool_size=(7,7), strides=(7,7),
        #                      border_mode='valid', dim_ordering=self.dim_ordering)(x)
        # x = Flatten()(x)
        # x = Dropout(0.40)(x)
        # output = Dense(self.output_shape, activation=self.output_act, init='zero', name='output')(x)
        # #self.Model = Model(input=[input_0, input_1], output=[aux_output1, aux_output2, output])
        # #model = Model(input=[input_0, input_1], output=[aux_output1, aux_output2, output])
        # ##self.output=output

        # Input
        # Do not include batch size in input shape
        init=self.init
        subsample=self.subsample
        activation=self.Activation
        input_0 = Input(shape=self.input_shape, dtype='float32', name='input_0') 
        x0 = input_0
        # dim_ordering is 'th' or 'tf' for channels at index 1 or at index 3
        x0 = Convolution2D(self.nb_filter, self.nb_row, self.nb_column,
                           init=self.init, activation=self.Activation, border_mode='same', 
                           subsample=subsample, dim_ordering='th', bias=True, 
                           W_regularizer=None, b_regularizer=None)(x0)

        input_1 = Input(shape=self.input_shape, dtype='float32', name='input_1') 
        x1 = input_1
        x1 = Convolution2D(self.nb_filter, self.nb_row, self.nb_column,
                           init=self.init, activation=self.Activation, border_mode='same', 
                           subsample=subsample, dim_ordering='th', bias=True, 
                           W_regularizer=None, b_regularizer=None)(x1)

        x = merge([x0, x1], mode='concat', concat_axis=1)
        x = Convolution2D(self.nb_filter, 1, 1, 
                          init=self.init, activation=self.Activation, border_mode='same', 
                          subsample=(1, 1), dim_ordering='th', bias=True, 
                          W_regularizer=None, b_regularizer=None)(x)

        x = Convolution2D(64, 7, 7,
                          init=self.init, activation=self.Activation, border_mode='same', 
                          subsample=(2,2), dim_ordering='th', bias=True, 
                          W_regularizer=None, b_regularizer=None)(x)
        x = ZeroPadding2D(padding=(1,1), dim_ordering='th')(x)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2),
                          border_mode='valid', dim_ordering='th')(x)

        # Convolution layer 2
        x = Convolution2D(64, 1, 1,
                          init=self.init, activation=self.Activation, border_mode='same', 
                          subsample=(1,1), dim_ordering='th', bias=True, 
                          W_regularizer=None, b_regularizer=None)(x)
        x = Convolution2D(192, 3, 3,
                          init=self.init, activation=self.Activation, border_mode='same', 
                          subsample=(1,1), dim_ordering='th', bias=True, 
                          W_regularizer=None, b_regularizer=None)(x)
        x = ZeroPadding2D(padding=(1,1), dim_ordering='th')(x)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2),
                          border_mode='valid', dim_ordering='th')(x)


        # Inception modules 3a-b
        x = InceptionModule(x, 64, 96, 128, 16, 32, 32, init, activation)
        x = InceptionModule(x, 128, 128, 192, 32, 96, 64, init, activation)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2),
                          border_mode='same', dim_ordering='th')(x)

        # Inception modules 4a-e
        x = InceptionModule(x, 192, 96, 208, 16, 48, 64, init, activation)
        aux_output1 = AuxiliaryClassifier(x, init=self.init, activation=self.Activation, output_shape=self.output_shape, output_act=self.output_act, output_name='aux_output1')
        x = InceptionModule(x, 160, 112, 224, 24, 64, 64, init, activation)
        x = InceptionModule(x, 128, 128, 256, 24, 64, 64, init, activation)
        x = InceptionModule(x, 112, 144, 288, 32, 64, 64, init, activation)
        aux_output2 = AuxiliaryClassifier(x, init=self.init, activation=self.Activation, output_shape=self.output_shape, output_act=self.output_act, output_name='aux_output2')
        x = InceptionModule(x, 256, 160, 320, 32, 128, 128, init, activation)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2),
                          border_mode='same', dim_ordering='th')(x)

        # Inception modules 5a-b
        x = InceptionModule(x, 256, 160, 320, 32, 128, 128, init, activation)
        x = InceptionModule(x, 384, 192, 384, 48, 128, 128, init, activation)

        # Average pooling layer
        # use mode to implement average pooling
        x = AveragePooling2D(pool_size=(7,7), strides=(7,7), 
                              border_mode='valid', dim_ordering='th')(x)
        x = Flatten()(x)
        x = Dropout(0.40)(x)

        # Fully connected layer
        output = Dense(self.output_shape, activation=self.output_act, init='zero', name='output')(x)

        # Model
        self.Model = Model(input=[input_0, input_1], output= [aux_output1, aux_output2, output])


