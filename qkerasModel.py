import h5py
import numpy as np
import argparse
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input, GlobalAveragePooling1D
#from dataForgeScripts.dataForge import N_FEAT, N_PART_PER_JET
from qkeras import *
from tensorflow.keras.regularizers import l1

from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule

import tensorflow_model_optimization as tfmot

N_FEAT = 14
N_PART_PER_JET = 10

def main(args):

    signalTrainFile = args.SignalTrainFile
    bkgTrainFile = args.BkgTrainFile
    sig_jetData_TrainFile = args.sig_jetData_TrainFile
    bkg_jetData_TrainFile = args.bkg_jetData_TrainFile

    print("Reading signal from " + signalTrainFile)
    print("Reading background from " + bkgTrainFile)
    print("Reading signal jet data from " + sig_jetData_TrainFile)
    print("Reading background jet data from " + bkg_jetData_TrainFile)

    with h5py.File(signalTrainFile, "r") as hf:
        dataset = hf["Training Data"][:]
    with h5py.File(bkgTrainFile, "r") as hf:
        datasetQCD = hf["Training Data"][:]
    with h5py.File(sig_jetData_TrainFile, "r") as hf:
        sampleData = hf["Sample Data"][:]
    with h5py.File(bkg_jetData_TrainFile, "r") as hf:
        sampleDataQCD = hf["Sample Data"][:]
    
    
    """" I am combining the features with jet data in order to shuffle all at 
         at once. This way, I will still have a 1-1 correspondance of jets and data
         after shuffling. """
    
    dataset = np.concatenate((dataset, datasetQCD))#Put datasets on top of one another
    sampleData = np.concatenate((sampleData,sampleDataQCD))
    fullData = np.concatenate((dataset, sampleData), axis=1)
    np.random.shuffle(fullData) #randomize QCD and Stop samples
    dataset = fullData[0:,0:141]
    sampleData = fullData[0:,141:]



    # Separate datasets into inputs and outputs, expand the dimensions of the inputs to be used with Conv1D layers
    X = dataset[:, 0 : len(dataset[0]) - 1]
    y = dataset[:, len(dataset[0]) - 1]
    X = X.reshape((X.shape[0], N_PART_PER_JET, N_FEAT))

    # Establish the sample weights
    thebins = np.linspace(0, 500, 20) # check for right range
    bkgPts = sampleData[y==0][:,0]
    sigPts = sampleData[y==1][:,0]
    bkg_counts, _ = np.histogram(bkgPts, bins=thebins)
    sig_counts, _ = np.histogram(sigPts, bins=thebins)
    total_bkg = len(bkgPts)
    total_sig  = len(sigPts)
    
    weights_pt = np.nan_to_num(sig_counts / bkg_counts, nan=total_sig / total_bkg)

    # Compile the network

    x = inputs = Input(shape=(10, 14), name="input_1")
    x = QConv1D(
        filters=10,
        kernel_size=1,
        strides=1,
        kernel_quantizer=quantized_bits(8, 3, alpha=1),
        bias_quantizer=quantized_bits(8, 3, alpha=1),
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(0.0001),
        bias_regularizer=l1(0.0001),
        name="q_conv1d",
    )(x)
    x = QActivation(activation=quantized_relu(8), name="q_activation")(x)
    x = QConv1D(
        filters=10,
        kernel_size=1,
        strides=1,
        kernel_quantizer=quantized_bits(8, 3, alpha=1),
        bias_quantizer=quantized_bits(8, 3, alpha=1),
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(0.0001),
        bias_regularizer=l1(0.0001),
        name="q_conv1d_1",
    )(x)
    x = QActivation(activation=quantized_relu(8), name="q_activation_1")(x)
    x = GlobalAveragePooling1D(name="global_average_pooling1d")(x)
    x = QDense(
        10,
        kernel_quantizer=quantized_bits(8, 3, alpha=1),
        bias_quantizer=quantized_bits(8, 3, alpha=1),
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(0.0001),
        bias_regularizer=l1(0.0001),
        name="q_dense",
    )(x)
    x = QActivation(activation=quantized_relu(8), name="q_activation_2")(x)
    x = QDense(
        1,
        kernel_quantizer=quantized_bits(8, 3, alpha=1),
        bias_quantizer=quantized_bits(8, 3, alpha=1),
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(0.0001),
        bias_regularizer=l1(0.0001),
        name="q_dense_1",
    )(x)
    outputs = Activation(activation="sigmoid", name="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs, name="model")


    #Pruning Step
    pruning_params = {
        "pruning_schedule":
            pruning_schedule.ConstantSparsity(0.75, begin_step=2000, frequency=100)
    }
    full_prune = True
    if full_prune:
        model = prune.prune_low_magnitude(model, **pruning_params)
    

    initial_learning_rate = 0.001
    lr_schedule = tensorflow.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    """The line below compiles the model with the learning schedule"""
    #model.compile(loss="binary_crossentropy", optimizer=tensorflow.keras.optimizers.Adam(
        #learning_rate=lr_schedule), metrics=["binary_accuracy"])

    """The line below uses the default Adam learning rate"""
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])


    # Add in the sample weights, 1-to-1 correspondence with training data
    # Sample weight of all signal events being equal to 1
    # Sample weight of all background events being equal to the sig/bkg ratio at that jet's pT
    weights = np.ones(len(y))
    pt_indicies = np.clip(np.digitize(sampleData[:,0], bins=thebins) - 1, 0, len(weights_pt) - 1)
    weights[y==0] = weights_pt[pt_indicies][y==0]


    # Train the network
    callbacks = [tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=10),
    pruning_callbacks.UpdatePruningStep()]
    
    history=model.fit(
        X,
        y,
        epochs=200,
        batch_size=50,
        verbose=2,
        sample_weight=np.asarray(weights),
        validation_split=0.20,
        callbacks=[callbacks],
    )
    plt.figure(figsize=(7,5), dpi=120)
    plt.plot(history.history['loss'], label = 'Train')
    plt.plot(history.history['val_loss'], label = 'Validation')
    plt.title('Model Loss', fontsize=25)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.savefig("qkModLoss.png")

    model = tfmot.sparsity.keras.strip_pruning(model)
    # Save the network
    model.save("qkL1JetTagModel.h5")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments")

    parser.add_argument("SignalTrainFile", type=str, help="input the signal train file")
    parser.add_argument("BkgTrainFile", type=str, help="input the signal train file")
    parser.add_argument("sig_jetData_TrainFile", type=str, help="input signal jet data of the form ...sampleData.h5")
    parser.add_argument("bkg_jetData_TrainFile", type=str, help="input the bkg jet data of form ...sampleDataQCD.h5 for example")

    args = parser.parse_args()
    main(args)
