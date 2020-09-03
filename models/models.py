# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:12:20 2019

@author: thakk
"""

# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K

######
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
######

from tensorflow.keras.models import  model_from_json

from tensorflow.keras.layers import  SeparableConv2D, UpSampling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Add
from tensorflow.keras import backend as K


class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itselfLS
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
 
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same",
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))
        
        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
 
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
 
        # return the constructed network architecture
        return model
    
class VGG16:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
#       chanDim = -1
 
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
#           chanDim = 1
        
        # CONV => RELU => POOL
#       model.add(Conv2D(input_shape=inputShape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
#       model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
#       model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#       model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
#       model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
#       model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#       model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
#       model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
#       model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
#       model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#       model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#       model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#       model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#       model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#       model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#       model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#       model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
#       model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
#        
#       model.add(Flatten())
#       model.add(Dense(units=4096,activation="relu"))
#       model.add(Dense(units=4096,activation="relu"))
#       model.add(Dense(units=classes, activation="softmax"))
        
        
        
        #############
        
        # Reference: http://agnesmustar.com/2017/05/25/build-vgg16-scratch-part-ii/
        
        # Conv Block 1
        model.add(Conv2D(64, (3, 3), input_shape=inputShape, activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # Conv Block 2
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # Conv Block 3
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # Conv Block 4
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # Conv Block 5
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # FC layers
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(classes, activation='softmax'))
                
        
        
        ############
        
 
        # return the constructed network architecture
        return model

########################
        
class InceptionV3:
    @staticmethod
    
    
    # function for creating a projected inception module
    def inception_module(Inpt_img, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
        # 1x1 conv
        conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(Inpt_img)
        # 3x3 conv
        conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(Inpt_img)
        conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
        # 5x5 conv
        conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(Inpt_img)
        conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
        # 3x3 max pooling
        pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(Inpt_img)
        pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
        # concatenate filters, assumes filters/channels last
        layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
        return layer_out
    
    
    def build(width, height, depth, classes):
        
#        inputShape = (height, width, depth)
    
        Inpt_img = Input(shape=(height, width, depth))
         
    
        # add inception block 1
        layer = InceptionV3.inception_module(Inpt_img, 64, 96, 128, 16, 32, 32)
        # add inception block 1
        layer = InceptionV3.inception_module(layer, 128, 128, 192, 32, 96, 64)
        # create model
        output = Flatten()(layer)
        
        out= Dense(classes, activation='softmax')(output)
        model = Model(inputs=Inpt_img, outputs=out)
        # summarize model
        print(model.summary())
        return model
    
#########################    
class Xception:
    
    def entry_flow(self,inputs) :
    
        x = Conv2D(32, (3,3), strides = 2, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
        x = Conv2D(64,3,padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
        previous_block_activation = x
    
        for size in [128, 256, 728] :
    
            x = Activation('relu')(x)
            x = SeparableConv2D(size, 3, padding='same')(x)
            x = BatchNormalization()(x)
    
            x = Activation('relu')(x)
            x = SeparableConv2D(size, 3, padding='same')(x)
            x = BatchNormalization()(x)
    
            x = MaxPooling2D(3, strides=2, padding='same')(x)
    
            residual = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation)
    
            x = Add()([x, residual])
            previous_block_activation = x
    
        return x
    
    
    def middle_flow(self,x, num_blocks=8) :
    
        previous_block_activation = x
    
        for _ in range(num_blocks) :
    
            x = Activation('relu')(x)
            x = SeparableConv2D(728, 3, padding='same')(x)
            x = BatchNormalization()(x)
    
            x = Activation('relu')(x)
            x = SeparableConv2D(728, 3, padding='same')(x)
            x = BatchNormalization()(x)
    
            x = Activation('relu')(x)
            x = SeparableConv2D(728, 3, padding='same')(x)
            x = BatchNormalization()(x)
    
            x = Add()([x, previous_block_activation])
            previous_block_activation = x
    
        return x
    
    
    def exit_flow(self,x) :
    
        previous_block_activation = x
    
        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)
    
        x = Activation('relu')(x)
        x = SeparableConv2D(1024, 3, padding='same')(x) 
        x = BatchNormalization()(x)
    
        x = MaxPooling2D(3, strides=2, padding='same')(x)
    
        residual = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation)
        x = Add()([x, residual])
    
        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)
    
        x = Activation('relu')(x)
        x = SeparableConv2D(1024, 3, padding='same')(x)
        x = BatchNormalization()(x)
    
        x = GlobalAveragePooling2D()(x)
    
        return x
          
    
    def build(self,width, height, depth, classes) :
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        #model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
    
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
            
        inputs = Input(shape=(height, width, depth))
        outputs = self.exit_flow(self.middle_flow(self.entry_flow(inputs)))
        outputs = Dense(classes, activation='softmax')(outputs)
        
        m = Model(inputs, outputs)
    
    
        #print(m.summary())
        return m
        
#Reference:https://towardsdatascience.com/implementing-a-resnet-model-from-scratch-971be7193718
class ResNet:

    def identity_block(self,X, f, filters, stage, block):
        """
        Implementation of the identity block.

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value. Need this later to add back to the main path.
        X_shortcut = X

        # First component of main path
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path (≈3 lines)
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path (≈2 lines)
        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        # Final step: Added shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def convolutional_block(self,X, f, filters, stage, block, s=2):
        """
        Implementation of the convolutional block.

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used

        Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value
        X_shortcut = X

        ##### MAIN PATH #####
        # First component of main path
        X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path (≈3 lines)
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path (≈2 lines)
        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
                   kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        ##### SHORTCUT PATH #### (≈2 lines)
        X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                            kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

        # Final step: Added shortcut value to main path, and passed it through a RELU activation (≈2 lines)
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X

    def ResNet50(self,input_shape=(64, 64, 3), classes=9):
        """
        Implementation of the popular ResNet50. Architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """

        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape)

        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)

        # Stage 1
        X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = self.convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='c')

        ### START CODE HERE ###

        # Stage 3 (≈4 lines)
        X = self.convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='d')

        # Stage 4 (≈6 lines)
        X = self.convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

        # Stage 5 (≈3 lines)
        X = self.convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

        # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        X = AveragePooling2D((2, 2), name="avg_pool")(X)

        ### END CODE HERE ###

        # output layer
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

        # Create model
        model = Model(inputs=X_input, outputs=X, name='ResNet50')

        return model



        

    

