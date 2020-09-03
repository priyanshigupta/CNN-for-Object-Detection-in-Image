# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:37:13 2019

@author: Priyanshi Gupta
"""
#python F:\IIS_Project\Project\train_v2.py --dataset dataset --model pokedex.model --labelbin lb.pickle --VGG16

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
####
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
####
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from models.models import SmallerVGGNet
from models.models import VGG16
from models.models import InceptionV3
from models.models import Xception 
#import pyimagesearch.inception_tf_v2 as Xcep 
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset (i.e., directory of images)")

ap.add_argument("-model", "--model", type=str, default="VGG16",
    help="name of pre-trained network to use")

ap.add_argument("-m", "--model_file", required=True,
    help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
    help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="path to output accuracy/loss plot")
#ap.add_argument("--VGGsmaller", help="execute VGGsmaller model",
#                    action="store_true")
#ap.add_argument("--VGG16", help="execute VGG16 model",
#                    action="store_true")
#ap.add_argument("--Resnet", help="execute Resnet model",
#                    action="store_true")
#ap.add_argument("--inception", help="execute Xception model",
#                    action="store_true")

args = vars(ap.parse_args())
supported_models = ["SmallerVGGNet","VGG16","InceptionV3","Xception"]

if args["model"] not in supported_models:
    raise AssertionError("The --model command line argument should be from 'SmallerVGGNet','VGG16','InceptionV3'")

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 10
INIT_LR = 1e-3
BS = 16
if args["model"] == "SmallerVGGNet":
    IMAGE_DIMS = (96, 96, 3)
if args["model"] == "VGG16":
    IMAGE_DIMS = (64, 64, 3)
if args["model"] == "InceptionV3":
    IMAGE_DIMS = (256, 256, 3)  
if args["model"] == "Xception":
    IMAGE_DIMS = (299, 299, 3)      
    
print("Img dms are:",IMAGE_DIMS)
 
# initialize the data and labels
data = []
labels = []
 
# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)


# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)
 
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)


# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
    data.nbytes / (1024 * 1000.0)))
 
# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
 
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")




# initialize the model
if args["model"] ==  "SmallerVGGNet":
    print("[INFO] compiling SmallerVGGNet model...")
    model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
        depth=IMAGE_DIMS[2], classes=len(lb.classes_))

if args["model"] == "VGG16":
    print("[INFO] compiling VGG16 model...")
    model = VGG16.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
        depth=IMAGE_DIMS[2], classes=len(lb.classes_))

if args["model"] == "InceptionV3":
    print("[INFO] compiling InceptionV3 model...")
    model = InceptionV3.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                              depth=IMAGE_DIMS[2], classes=len(lb.classes_))
    

if args["model"] == "Xception":
    print("[INFO] compiling InceptionV3 model...")
    Xcep=Xception()
    model = Xcep.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                              depth=IMAGE_DIMS[2], classes=len(lb.classes_))    
        



opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

#######################################################################


checkpoint = ModelCheckpoint(args["model_file"], monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=1, mode='auto')

####################################################################### 

# train the network
print("[INFO] training " + str(args["model"]) + " network...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),steps_per_epoch=len(trainX) // BS,epochs=EPOCHS, verbose=1, callbacks = [checkpoint])

# save the model to disk
#print("[INFO] serializing network...")
#model.save(args["model"])
 
# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()



# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy for "+str(args["model"]))
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])