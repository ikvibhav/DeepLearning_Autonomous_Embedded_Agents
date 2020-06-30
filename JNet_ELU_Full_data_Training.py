import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split  
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
import argparse
import os
import matplotlib.pyplot as plt
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 
#from keras.backend import tf as ktf
from PIL import Image
#for debugging, allows for reproducible (deterministic) results 
np.random.seed(0)

#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 65, 320, 3 #66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[0:-25, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    #image = rgb2yuv(image)
    return image


def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    #reads CSV file into a single dataframe variable
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    #yay dataframes, we can select rows and columns by their names
    #we'll store the camera images as our input data
    X = data_df[['center', 'left', 'right']].values
    #and our steering commands as our output data
    y = data_df['steering'].values

    #now we can split the data into a training (80), testing(20), and validation set
    #thanks scikit learn
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    #IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 65, 320, 3

    #INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

    #X_train = X_train.reshape(X_train, (IMAGE_HEIGHT, IMAGE_WIDTH))
    #X_valid = X_valid.reshape(X_valid, (IMAGE_HEIGHT, IMAGE_WIDTH))
    #plt.imshow(Image.open(X[250,1]))
    #plt.show()
    #print(Image.open(X[250,1]))
    #print(X_train.shape)
    image_train = np.empty([X_train.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    #steers = np.empty(batch_size)
    #steers_train = np.empty(y_train.shape[0])
    image_val = np.empty([X_valid.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    #steers_val = np.empty(y_valid.shape[0])
    #dataset_size = X_train[0]
    i = 0
    for index in np.random.permutation(X_train.shape[0]):
        center, left, right = X_train[index]
        image = load_image(args.data_dir, center)
        image_train[i] = preprocess(image)
        i=i+1
        if i == X_train.shape[0]:
            break

    i = 0
    for index in np.random.permutation(X_valid.shape[0]):
        center, left, right = X_valid[index]
        image = load_image(args.data_dir, center)
        image_val[i] = preprocess(image)
        i=i+1
        if i == X_valid.shape[0]:
            break                        
    #print(image_train.shape)
    print(image_train.shape)
    print(image_val.shape)
    return image_train, image_val, y_train, y_valid

    #return X_train, X_valid, y_train, y_valid

def build_model(args):
    model = Sequential()
    model.add(Lambda(lambda x: x/255-0.5, input_shape=INPUT_SHAPE))  #Image-Normalisation layer - Avoids Saturation and make gradients work better
    model.add(Conv2D(16, 5, 5, activation='elu'))   
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, 5, 5, activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, 5, 5, activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(args.keep_prob))                 #For deeper network, if overfitting increasing the keep dropout more than 0.5 (To check)
    model.add(Flatten())
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.summary()

    return model

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    #Saves the model after every epoch.
    #quantity to monitor, verbosity i.e logging mode (0 or 1), 
    #if save_best_only is true the latest best model according to the quantity monitored will not be overwritten.
    #mode: one of {auto, min, max}. If save_best_only=True, the decision to overwrite the current save file is
    # made based on either the maximization or the minimization of the monitored quantity. For val_acc, 
    #this should be max, for val_loss this should be min, etc. In auto mode, the direction is automatically
    # inferred from the name of the monitored quantity.
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='auto')
    #calculate the difference between expected steering angle and actual steering angle
    #square the difference
    #add up all those differences for as many data points as we have
    #divide by the number of them
    #that value is our mean squared error! this is what we want to minimize via
    #gradient descent
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])

    #Fits the model on data generated batch-by-batch by a Python generator.

    #The generator is run in parallel to the model, for efficiency. 
    #For instance, this allows you to do real-time data augmentation on images on CPU in 
    #parallel to training your model on GPU.
    #so we reshape our data into their appropriate batches and train our model simulatenously
    
    history = model.fit(X_train, y_train, batch_size=args.batch_size, nb_epoch=args.nb_epoch,
                    validation_data=(X_valid, y_valid), callbacks=[checkpoint])

    #history = model.fit(
    #x=X_train, y=y_train, batch_size=None, epochs=1, validation_data=None, shuffle=True, class_weight=None,
    #sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    #validation_steps=None, validation_batch_size=None, validation_freq=1,
    #max_queue_size=10, workers=1, use_multiprocessing=False
    #)
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    #generates a plot of the accuracy and loss values
    epochs_range = range(args.nb_epoch)

    plt.figure(figsize=(16, 16))
    plt.suptitle('JNet Model')
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(' Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('JNet_model_plots_training.png')
    plt.show()


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)       #10
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=100)       #40
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=46500)     #15500
    #parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #load data
    data = load_data(args)
    #build model
    model = build_model(args)
    #train model on data, it saves as model.h5 
    
    train_model(model, args, *data)
    #print(history.history.keys())
    #Check https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/

if __name__ == '__main__':
    main()

