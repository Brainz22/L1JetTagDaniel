from numpy import loadtxt
from numpy import expand_dims
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import sys, os, numpy

import tensorflow
from sklearn.model_selection import train_test_split

from qkeras.utils import load_qmodel

#os.environ['PATH'] = '/data/therwig/Xilinx/Vivado/2019.2/bin:' + os.environ['PATH']
os.environ['PATH'] = '/data/software/xilinx/Vivado/2020.1/bin:' + os.environ['PATH']
#os.environ['PATH'] = '/data/software/xilinx/Vitis_HLS/2023.2/bin/' + os.environ['PATH']

model = load_qmodel("qkL1JetTagModel.h5")

#Conver model to HLS
import hls4ml
config = hls4ml.utils.config_from_keras_model(model, granularity='name')
print("-----------------------------------")

config['LayerName']['q_conv1d']['ReuseFactor'] = 1
config['LayerName']['q_conv1d_1']['ReuseFactor'] = 1
config["LayerName"]["q_conv1d"]["ConvImplementation"] = "Pointwise"
config["LayerName"]["q_conv1d_1"]["ConvImplementation"] = "Pointwise"

config["LayerName"]["input_1"]["Precision"] = "fixed<16,6>"
config['LayerName']['q_conv1d']["Precision"] = "fixed<16,6>"
config["LayerName"]["q_activation"]["Precision"] = "fixed<16,6>"
config['LayerName']['q_conv1d_1']["Precision"] = "fixed<16,6>"
config["LayerName"]["q_activation_1"]["Precision"] = "fixed<16,6>"
config["LayerName"]["q_activation_2"]["Precision"] = "fixed<16,6>"

#For Tracing
for layer in config['LayerName'].keys():
    print('Enable tracing for layer:', layer)
    config['LayerName'][layer]['Trace'] = True

hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=config,
                                                       output_dir='qkmodel/hls4ml_prj',
                                                       part='xcvu13p-flga2577-2-e')

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=os.getcwd() + "/LayerTraces/qkmodel.png")

#Compile model, no need to convert if we are plotting performance
hls_model.compile()

# Handle Data: 
with h5py.File("/home/users/russelld/L1JetTagDaniel/hls4mlModifications/10-08-23/02-02_datasets/M_LLP_30_ctau_10/newTestDatapt20_vDter_Signal_Only.h5", "r") as hf:
    dataset = hf["Testing Data"][:]
dataset = dataset[:, 0:141]
with h5py.File("/home/users/russelld/L1JetTagDaniel/backgroundQCD/testingDatapt30QCD.h5", "r") as hf:
    datasetQCD = hf["Testing Data"][:]
    
dataset = np.concatenate((dataset,datasetQCD)) #Stacking datasets on top of another
np.random.shuffle(dataset) #shuffling rows
   
N_PART_PER_JET = 10
N_FEAT = 14
A = dataset[:, 0 : len(dataset[0]) - 1]
b = dataset[:, len(dataset[0]) - 1]
#A = expand_dims(A, axis=3)
A = A.reshape((A.shape[0], N_PART_PER_JET, N_FEAT))
X_test = np.ascontiguousarray(A)

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt

Ab_pred_qkeras = model.predict(A).ravel()
Ab_pred_hls_qkeras = hls_model.predict(X_test).ravel()

fpr_Ab_qkeras, tpr_Ab_qkeras, thresholds_Ab_qkeras = roc_curve(b, Ab_pred_qkeras)
auc_Ab_qkeras = auc(fpr_Ab_qkeras, tpr_Ab_qkeras)

fpr_Ab_hls, tpr_Ab_hls, thresholds_Ab_hls = roc_curve(b, Ab_pred_hls_qkeras)
auc_Ab_hls = auc(fpr_Ab_hls, tpr_Ab_hls)


plt.plot(fpr_Ab_qkeras, tpr_Ab_qkeras, label=" qkeras AUC={:.3f}, M_LLP_30_ctau_10".format(auc_Ab_qkeras))
plt.plot(fpr_Ab_hls, tpr_Ab_hls, "--" ,label=" HLS AUC={:.3f}".format(auc_Ab_hls))

plt.xlabel("Background Efficiency", fontsize=16)
plt.ylabel("Signal Efficiency", fontsize=16)
#plt.axvline(x=0.01, ymin=0, ymax=0.59, color="red")
#plt.axhline(y=0.6, xmin=0, xmax=0.573, color="red")
plt.title("L1 LLP Tag Qk Model ROC Curve", fontsize=16, weight="bold")
plt.legend(loc="best")
plt.xscale("log")
plt.grid(True)
plt.savefig("HLS_qk_ROCCurve.pdf")

#Layer Tracing

import hls4ml.model.profiling


y_hls, hls4ml_trace = hls_model.trace(X_test)
keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, X_test)

for LAYER in hls4ml_trace.keys():
    plt.figure()
    plt.scatter(hls4ml_trace[LAYER].flatten(), keras_trace[LAYER].flatten())
    min_x = min(np.amin(hls4ml_trace[LAYER]), np.amin(keras_trace[LAYER]))
    max_x = max(np.amax(hls4ml_trace[LAYER]), np.amax(keras_trace[LAYER]))
    plt.plot([min_x, max_x], [min_x, max_x], c='gray')
    plt.xlabel('hls4ml {}'.format(LAYER))
    #plt.xlabel('hls4ml {}'.format(LAYER))
    plt.ylabel('QKeras {}'.format(LAYER))
    print(os.getcwd() + f'/LayerTraces/profiling_{LAYER}.png')
    plt.savefig(os.getcwd() + f'/LayerTraces/profiling_{LAYER}.png')   

hls_model.build(csim=False)