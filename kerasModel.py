import h5py
import argparse
import numpy as np
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Input, GlobalAveragePooling1D
from dataForgeScripts.dataForge import N_FEAT, N_PART_PER_JET

from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule

import tensorflow_model_optimization as tfmot

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

    thebins = np.linspace(0, 200, 100)
    bkgPts = []
    sigPts = []
    for i in range(len(sampleData)):
        if y[i] == 1:
            sigPts.append(sampleData[i][0])
        if y[i] == 0:
            bkgPts.append(sampleData[i][0])
    bkg_counts, binsbkg = np.histogram(bkgPts, bins=thebins)
    sig_counts, binssig = np.histogram(sigPts, bins=thebins)
    a = []
    for i in range(len(bkg_counts)):
        tempSig = float(sig_counts[i])
        tempBkg = float(bkg_counts[i])
        if tempBkg != 0:
            a.append(tempSig / tempBkg)
        if tempBkg == 0:
            a.append(0)
# Normalize the sample weights above a certain pT
    for i in range(42, len(a)):
        a[i] = 0.7

    

# Compile the network
    x = inputs = Input(shape=(N_PART_PER_JET, N_FEAT))
    x = Conv1D(
        filters=50,
        kernel_size=4,
        strides=2,
        activation="relu",
    )(x)
    x = Conv1D(filters=50, kernel_size=4,strides=1, activation="relu")(x)
#x = GlobalAveragePooling1D()(x)
    x = Flatten()(x)
    x = Dense(50, activation="relu")(x)
    x = Dense(10, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

#Pruning Step
    pruning_params = {
        "pruning_schedule":
            pruning_schedule.ConstantSparsity(0.75, begin_step=2000, frequency=100)
    }
    full_prune = True
    if full_prune:
        model = prune.prune_low_magnitude(model, **pruning_params)
    


    print('\n')
    model.summary()
    print('\n')

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
    weights = []
    for i in range(len(sampleData)):
        if y[i] == 1:
            weights.append(1)
        if y[i] == 0:
            jetPt = sampleData[i][0]
            tempPt = int(jetPt / 2)
            if tempPt > 98:
                tempPt = 98
            weights.append(a[tempPt])

# Train the network
    callbacks = [tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=10),
                 pruning_callbacks.UpdatePruningStep(),
    ]
                 
    history=model.fit(
        X,
        y,
        epochs=50,
        batch_size=50,
        verbose=2,
        sample_weight=np.asarray(weights),
        validation_split=0.20,
        callbacks=callbacks,
    )
    plt.figure(figsize=(7,5), dpi=120)
    plt.plot(history.history['loss'], label = 'Train')
    plt.plot(history.history['val_loss'], label = 'Validation')
    plt.title('Model Loss', fontsize=25)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.savefig("modelLoss.png")


    #Strip pruning before saving
    model = tfmot.sparsity.keras.strip_pruning(model)

    # Save the network
    model.save("L1JetTagModel.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments")

    parser.add_argument("SignalTrainFile", type=str, help="input the signal train file")
    parser.add_argument("BkgTrainFile", type=str, help="input the signal train file")
    parser.add_argument("sig_jetData_TrainFile", type=str, help="input signal jet data of the form ...sampleData.h5")
    parser.add_argument("bkg_jetData_TrainFile", type=str, help="input the bkg jet data of form ...sampleDataQCD.h5 for example")

    args = parser.parse_args()
    main(args)
