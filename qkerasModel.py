import h5py, os
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
from tensorflow.keras.utils import plot_model

import tensorflow_model_optimization as tfmot
from sklearn.preprocessing import MinMaxScaler

from inputFixer import add_ip

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
        dataset = hf["jet_constituents"][:]
    with h5py.File(bkgTrainFile, "r") as hf:
        datasetQCD = hf["jet_constituents"][:]
    with h5py.File(sig_jetData_TrainFile, "r") as hf:
        sampleData = hf["train_jet_data"][:]
    with h5py.File(bkg_jetData_TrainFile, "r") as hf:
        sampleDataQCD = hf["train_jet_data"][:]
    
    
    """" I am combining the features with jet data in order to shuffle all at 
         at once. This way, I will still have a 1-1 correspondance of jets and data
         after shuffling. """
    
    dataset = np.concatenate((dataset, datasetQCD))#Put datasets on top of one another
    sampleData = np.concatenate((sampleData,sampleDataQCD))
    fullData = np.concatenate((dataset, sampleData), axis=1)
    np.random.shuffle(fullData) #randomize QCD and Stop samples
    dataset = fullData[0:,0:141]
    sampleData = fullData[0:,141:]
    #LLPfeats = fullData[0:,142:146]
    #sampleData = fullData[0:,146:]



    # Separate datasets into inputs and outputs, expand the dimensions of the inputs to be used with Conv1D layers
    X = dataset[:, 0 : len(dataset[0]) - 1]
    y = dataset[:, len(dataset[0]) - 1]
    X = X.reshape((X.shape[0], N_PART_PER_JET, N_FEAT))

    X = add_ip(X) #add ip as feature instead of separate dx and dy features

    
    
    #plot kinematics
    from plotting.kinematics_plotter import kinematics
    #Normalize impact parameter?
    normalizeIPs = False # Knob to say if I want to normalize IPs.

    if max(X[:, :, 8].ravel()) < 2.0:
        norm_b4 = True
    else: 
        print("\nImpact parameter was not normalized beforehand.\n")
        norm_b4 = False

    if norm_b4:
        tag = "separateNorm/sepNorm_train"
        kinematics(X, sampleData, y, "4b_4c_4u", tag)
    elif normalizeIPs:
        tag = "b4train_Norm/Norm_b4train"
        kinematics(X, sampleData, y, "4b_4c_4u", "b4train_Norm/unNorm_train")
    else:
        tag = "noNorm/noNorm_train"
        kinematics(X, sampleData, y, "4b_4c_4u", tag)


    # Actual normalization performed below if normalizeIPs=True. 
    # If norm_b4=True, I just plot kinematics since normalization happened beforehand.

    if norm_b4:
        print("\nImpact parameter was normalized beforehand.\n")

    elif normalizeIPs: 
        print("\nNormalizing Impact parameter done here.\n")

        scaler = MinMaxScaler(feature_range=(-1, 1))
    
        temp_dz = scaler.fit_transform([[dz] for dz in X[:, :, 8].ravel()])
        X[:, :, 8] = temp_dz.reshape(X[:,: ,8].shape)
        temp_dx = scaler.fit_transform([[dx] for dx in X[:, :, 9].ravel()])
        X[:, :, 9] = temp_dx.reshape(X[:,: ,9].shape)
        temp_dy = scaler.fit_transform([[dy] for dy in X[:, :, 10].ravel()])
        X[:, :, 10] = temp_dy.reshape(X[:, : ,10].shape)


        #plot kinematics
        #kinematics(X, sampleData, y, "4b_4c_4u", tag)
    else:
        print("\nDecided not to normalize impact parameter. \n")
        tag = "noNorm/noNorm_train"
        #kinematics(X, sampleData, y, "4b_4c_4u", tag)



    # Establish the sample weights
    thebins = np.linspace(0, max(sampleData[:, 0]), 60) # check for right range
    bkgPts = sampleData[y==0][:,0]
    sigPts = sampleData[y==1][:,0]
    bkg_counts, _ = np.histogram(bkgPts, bins=thebins)
    sig_counts, _ = np.histogram(sigPts, bins=thebins)
    total_bkg = len(bkgPts)
    total_sig  = len(sigPts)
    
    weights_pt = np.nan_to_num(sig_counts / bkg_counts, nan=total_sig / total_bkg)

    #Plot for understanding weights

    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )
    fig.subplots_adjust(hspace=0.05)

    # Main panel
    ax_main.step(thebins[:-1], sig_counts, where="post", label="Signal")
    ax_main.step(thebins[:-1], bkg_counts, where="post", label="Bkg")
    ax_main.set_ylabel("Events")
    ax_main.legend()

    # Ratio panel
    ax_ratio.step(thebins[:-1], weights_pt, where="post", color="black")
    ax_ratio.axhline(1.0, color="red", linestyle="--", linewidth=1)
    ax_ratio.set_ylabel("weights")
    ax_ratio.set_xlabel(r"$P_T$")
    fig.savefig(os.getcwd() + f"/{tag}_13_pt_weights.png", bbox_inches="tight", dpi=150 )


    # Compile the network

    x = inputs = Input(shape=(10, 13), name="input_1")
    #x = QActivation(activation=quantized_tanh(10), name="q_tanh")(x)
    #x = QBatchNormalization(axis = -1 , name="batch_norm" )(x)
    x = QActivation(activation=quantized_bits(12, 6, alpha=1), name="q_input")(x)

    x = QConv1D(
        filters=10,
        kernel_size=1,
        strides=1,
        kernel_quantizer=quantized_bits(10, 4, alpha=1),
        bias_quantizer=quantized_bits(10, 4, alpha=1),
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(0.0001),
        bias_regularizer=l1(0.0001),
        name="q_conv1d",
    )(x)
    x = QActivation(activation=quantized_relu(10, 5) , name="q_activation")(x)
    #x = QActivation(activation=quantized_tanh(10), name="q_tanh_1")(x)
    x = QConv1D(
        filters=10,
        kernel_size=1,
        strides=1,
        kernel_quantizer=quantized_bits(10, 4, alpha=1),
        bias_quantizer=quantized_bits(10, 4, alpha=1),
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(0.0001),
        bias_regularizer=l1(0.0001),
        name="q_conv1d_1",
    )(x)
    x = QActivation(activation=quantized_relu(10, 5), name="q_activation_1")(x)

    x = GlobalAveragePooling1D(name="global_average_pooling1d")(x)

    x = QDense(
        10,
        kernel_quantizer=quantized_bits(10, 4, alpha=1),
        bias_quantizer=quantized_bits(10, 4, alpha=1),
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(0.0001),
        bias_regularizer=l1(0.0001),
        name="q_dense",
    )(x)
    x = QActivation(activation=quantized_relu(10, 5), name="q_activation_2")(x)
    outputs = QDense(
        1,
        kernel_quantizer=quantized_bits(10, 4, alpha=1),
        bias_quantizer=quantized_bits(10, 4, alpha=1),
        kernel_initializer="lecun_uniform",
        kernel_regularizer=l1(0.0001),
        bias_regularizer=l1(0.0001),
        name="q_dense_1",
    )(x)
    #outputs = Activation(activation="sigmoid", name="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs, name="model")

    plot_model(model, to_file=os.getcwd() + f"/{tag}_13_model.png", show_shapes=True, show_layer_names=True )


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
    # model.compile(loss="binary_crossentropy", optimizer=tensorflow.keras.optimizers.Adam(
    #     learning_rate=lr_schedule), metrics=["binary_accuracy"])

    """The line below uses the default Adam learning rate"""
    #model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"])
    model.compile(loss=tensorflow.keras.losses.BinaryCrossentropy(from_logits=True, name="binary_crossentropy"), 
                  optimizer="adam", 
                  metrics=["binary_accuracy"],
                  weighted_metrics=[tensorflow.keras.metrics.AUC(name="auc")])
    

    """Weights add more importance to low pt jets, since we assign the ratio of signal_count / bkg_count in that pt bin as the weight for all jets in such a bin.
    This is because we want to make sure the model learns to classify low pt jets well, since they are more common and more difficult to classify."""
    # Add in the sample weights, 1-to-1 correspondence with training data
    # Sample weight of all signal events being equal to 1
    # Sample weight of all background events being equal to the sig/bkg ratio at that jet's pT
    weights = np.ones(len(y))
    pt_indicies = np.clip(np.digitize(sampleData[:,0], bins=thebins) - 1, 0, len(weights_pt) - 1)
    weights[y==0] = weights_pt[pt_indicies][y==0]
    
    
    plt.figure()
    plt.hist(weights, bins = 51 )
    plt.xlabel("Weights")
    plt.savefig("{}_13_LossFromLogits_weights.png".format( tag))

    np.save("{}_13_qkmodelWeights.npy".format(tag), weights)
    np.save("{}_13_ptRange.npy".format( tag), sampleData[:,0])


    # Train the network
    callbacks = [tensorflow.keras.callbacks.EarlyStopping(monitor="val_loss", verbose=1, patience=65),
                pruning_callbacks.UpdatePruningStep(),
                tensorflow.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=15, verbose=1 ),
                tensorflow.keras.callbacks.ModelCheckpoint(filepath=os.getcwd() + "/{}_epochs_{{epoch}}_best_13qkmodel.h5".format(tag), monitor="val_loss", save_best_only=True)
                ]



    #training
    
    X = add_ip(X) #add ip as feature instead of separate dx and dy features

    print("=========================================")
    print("Starting training...")
    print("Inputs shape: ", X.shape)
    print("=========================================")
    
    history=model.fit(
        X,
        y,
        epochs=500,
        batch_size=50,
        verbose=2,
        sample_weight=np.asarray(weights),
        validation_split=0.20,
        callbacks=[callbacks],
    )

    print("Training finished. Keys of history object: " + str(history.history.keys()))

    plt.figure(figsize=(7,5), dpi=120)
    plt.plot(history.history['loss'], label = 'Train')
    plt.plot(history.history['val_loss'], label = 'Validation')
    plt.title('Model Loss', fontsize=25)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='best')
    plt.tight_layout()
    #plt.savefig(str(args.llpType) + "/qkLoss{}.pdf".format(str(args.llpType)) , dpi=120 )
    plt.savefig(os.getcwd() + "/{}_epochs_{epochs}_13qkLoss_LossFromLogits.pdf".format(tag, epochs=history.epoch[-1]) , dpi=120 )

    plt.figure(figsize=(7,5), dpi=120)
    plt.plot(history.history['binary_accuracy'], label = 'Train')
    plt.plot(history.history['val_binary_accuracy'], label = 'Validation')
    plt.title('Model Accuracy', fontsize=25)
    plt.ylabel('BinaryAccuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.tight_layout()
    #plt.savefig(str(args.llpType) + "/qkLoss{}.pdf".format(str(args.llpType)) , dpi=120 )
    plt.savefig(os.getcwd() + "/{}_epochs_{epochs}_13qkAccuracy_LossFromLogits.pdf".format(tag, epochs=history.epoch[-1]) , dpi=120 )

    plt.figure(figsize=(7,5), dpi=120)
    plt.plot(history.history['lr'], label = 'Train')
    #plt.plot(history.history['val_binary_accuracy'], label = 'Validation')
    plt.title('Model Accuracy', fontsize=25)
    plt.ylabel('BinaryAccuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    plt.tight_layout()
    #plt.savefig(str(args.llpType) + "/qkLoss{}.pdf".format(str(args.llpType)) , dpi=120 )
    plt.savefig(os.getcwd() + "/{}_epochs_{epochs}_13qkLR_LossFromLogits.pdf".format(tag, epochs=history.epoch[-1]) , dpi=120 )

    model = tfmot.sparsity.keras.strip_pruning(model)
    # Save the network
    #model.save(str(args.llpType) + "/"+ str(args.llpType)+ "qkL1JetTagModel.h5")
    model.save( os.getcwd() + "/{}_epochs_{epochs}_13qkL1JetTagModel_LossFromLogits.h5".format(tag, epochs=history.epoch[-1]) )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments")

    parser.add_argument("SignalTrainFile", type=str, help="input the signal train file")
    parser.add_argument("BkgTrainFile", type=str, help="input the background train file")
    parser.add_argument("sig_jetData_TrainFile", type=str, help="input signal jet data of the form ...sampleData.h5")
    parser.add_argument("bkg_jetData_TrainFile", type=str, help="input the bkg jet data of form ...sampleDataQCD.h5 for example")
    parser.add_argument("llpType", type=str, help="Folder name/path to save files for Specific LLPs.")

    args = parser.parse_args()
    main(args)
