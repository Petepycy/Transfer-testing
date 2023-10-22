import os
import random
import shutil
import json
import numpy as np

from argparse import ArgumentParser
from tqdm import tqdm
from math import ceil
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Model

# Running comand: python3 poc.v2.py --testedDataShare x [--> where x is percentage of added class, value in range: (0;1>  <--]

Experiment_Name= "..." #Is added to the name of result direcory

USED_GPU="GPU:0" #Is passed later for training and testing

# Default Classes names:
DEFAULT_CLASS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship"]

# TT Classes names:
TT_CLASS = ["truck_TT"]
#Add more classes if needed

BATCH_SIZE = 64
INPUT_SHAPE = (32,32,3)
NUM_CLASSES = 10
EPOCHS = 100


def load_and_combine_datasets(dbPath, full_data_directories, cut_data_directories, data_percentage):
    # Create empty lists to store datasets
    full_data_datasets = []
    cut_data_datasets = []
    specific_directory_path = os.path.join(dbPath, "train")

    try:
        # Load datasets from full_data_directories
        for directory in full_data_directories:
            dataset = tf.keras.utils.image_dataset_from_directory(
                os.path.join(specific_directory_path, directory),
                labels='inferred',
                label_mode='categorical',
                batch_size=BATCH_SIZE,
                image_size=(32, 32),
                shuffle=False
            )
            full_data_datasets.append(dataset)

        # Load datasets from cut_data_directories and apply data_percentage
        for directory in cut_data_directories:
            dataset = tf.keras.utils.image_dataset_from_directory(
                os.path.join(specific_directory_path, directory),
                labels='inferred',
                label_mode='categorical',
                batch_size=BATCH_SIZE,
                image_size=(32, 32),
                shuffle=False
            )
            data_size = int(len(dataset) * data_percentage)
            dataset = dataset.take(data_size)
            cut_data_datasets.append(dataset)

        # Concatenate datasets from all directories
        combined_dataset = tf.data.Dataset.concatenate(*full_data_datasets)
        combined_dataset = tf.data.Dataset.concatenate([combined_dataset] + cut_data_datasets)

        combined_dataset = combined_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return combined_dataset

    except Exception as e:
        print(f"An error occurred while loading datasets: {str(e)}")
        return None

def resize_image(image, label):

    # Resize the image to the desired dimensions:

    resized_image = tf.image.resize(image, (32, 32))
    return resized_image, label

def getLoaders(dbPath: str, data_percentage=1.0):

    #Defining paths to the directories:

    trainDir = os.path.join(dbPath, "train")
    valDir = os.path.join(dbPath, "val")
    testDirFull = os.path.join(dbPath, "testFull")
    testDir = os.path.join(dbPath, "test")

    #Creating datasets and resizing images

    # train_dataset = tf.keras.utils.image_dataset_from_directory(
    #     trainDir,
    #     labels='inferred',
    #     label_mode='categorical',
    #     batch_size=BATCH_SIZE,
    #     image_size=(32, 32),
    #     shuffle=True,
    #     seed=321
    # )
    train_dataset = load_and_combine_datasets(trainDir, DEFAULT_CLASS, TT_CLASS, data_percentage)

    #train_dataset = train_dataset.map(resize_image)

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        valDir,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=(32, 32),
        shuffle=False
    )

    #val_dataset = val_dataset.map(resize_image)

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        testDir,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=(32, 32),
        shuffle=False
    )

    #test_dataset = test_dataset.map(resize_image)

    test_dataset_full = tf.keras.utils.image_dataset_from_directory(
        testDirFull,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=(32, 32),
        shuffle=False
    )

    #test_dataset_full = test_dataset_full.map(resize_image)
    val_subset_size = int(len(valDataset) * data_percentage) 
    valDataset = valDataset.take(val_subset_size)

    #WHAT IS THAT???

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset_full = test_dataset_full.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset, test_dataset_full

def getModel(inputShape: Tuple, numClasses: int):

    #Definig model architecture: 

    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=inputShape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (5, 5), padding="same", activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), padding="same", activation="relu"), 
        Flatten(),
        Dense(64, activation="relu"),
        Dense(numClasses, activation="softmax")

    ])

    return model

if __name__ == "__main__":

    random.seed(2137)

    #parsing arguments

    parser = ArgumentParser()

    parser.add_argument('--testedDataShare', dest='testedDataShare', type=float, default=0.1,
                        help="what percentage of tested subset should be added to train and val splits")
    
    parser.add_argument("--dbPath", dest="dbPath", type=str, default="Transfer_testing_db",
                        help="Path to dir with raw subset we are interested in")
    
    args = parser.parse_args()

    #load datasets

    # trainDataset, valDataset, testDataset, testDatasetFull = getLoaders(os.path.join(args.dbPath, "data"))
    trainDataset, valDataset, testDataset, testDatasetFull = getLoaders(os.path.join(args.dbPath, "data"), data_percentage=args.testedDataShare)

    # # Calculate the number of elements to select from the validation and test datasets
    # val_subset_size = int(len(valDataset) * args.testedDataShare) 
    # test_subset_size = int(len(testDataset) * args.testedDataShare)

    # # Create subsets of the validation and test datasets for Transfer Testing
    # valDataset = valDataset.take(val_subset_size)
    # testDataset = testDataset.take(test_subset_size)

    model = getModel(INPUT_SHAPE, NUM_CLASSES)


    modelsPath = os.path.join(os.getcwd(), "models")
    if os.path.exists(modelsPath):
        shutil.rmtree(modelsPath)
    os.makedirs(modelsPath)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=[tf.keras.metrics.F1Score(average="macro")])
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(modelsPath, 'model_epoch{epoch:02d}.h5'),
        monitor='val_f1score',
        save_best_only=False,
        mode='max',
        verbose=1
    )

    #Training and Validation

    epochs = EPOCHS
    with tf.device(USED_GPU):
        history = model.fit(trainDataset, epochs=epochs, validation_data=testDatasetFull, callbacks=[checkpoint_callback])

    train_f1_scores = history.history['f1_score']
    val_f1_scores = history.history['val_f1_score']

    best_epoch = np.argmax(val_f1_scores)
    best_model = tf.keras.models.load_model(os.path.join(modelsPath, f'model_epoch{best_epoch:02d}.h5'))

    #Testing with TT test set and full testset

    with tf.device(USED_GPU):
        test_loss_TT, test_f1_score_TT = best_model.evaluate(valDataset)
        #test_loss_full, test_f1_score_full = best_model.evaluate(testDatasetFull)

    #Create metrics directory

    metricsPath = os.path.join(os.getcwd(), "results ", Experiment_Name)
    os.makedirs(metricsPath, exist_ok=True)

    #Saving metrics to the directory:

    saveDict = {
        "train_f1_score": round(train_f1_scores[best_epoch],3),
        "val_f1_score": round(val_f1_scores[best_epoch],3),
        "test_f1_score_TT": round(test_f1_score_TT,3),
        "test_loss_TT": round(test_loss_TT,3),
        #"test_f1_score_full": round(test_f1_score_full,3),
        #"test_loss_full": round(test_loss_full,3)
    }

    jsonPath = os.path.join(metricsPath, f"testSplit{args.testedDataShare}.json")
    if os.path.exists(jsonPath):
        os.remove(jsonPath)

    with open(jsonPath, 'w') as f:
        json.dump(saveDict, f, indent=2)