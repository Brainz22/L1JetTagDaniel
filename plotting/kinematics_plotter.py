import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import mplhep as hep
plt.style.use(hep.style.ROOT)

N_PART_PER_JET = 10
N_FEAT = 14

def kinematics(part_data, jet_data, labels, llpType, test_train):
    llpFolder = llpType
    method = test_train

    signal_parts = part_data[labels==1]
    signalJet_data = jet_data[labels==1]
    
    bkg_parts = part_data[labels==0]
    bkgJet_data = jet_data[labels==0]

    #Extracting bkg information
    bkg_dz = bkg_parts[:, :,  8].ravel()
    bkg_dxdy = bkg_parts[:, :,  9].ravel()
    #bkg_dy = bkg_parts[:, :, 10].ravel()
    bkg_pt = bkg_parts[:, :, 10].ravel()
    bkg_eta = bkg_parts[:, :, 11].ravel()
    bkg_phi = bkg_parts[:, :, 12].ravel()
    bkgDF = pd.DataFrame({"dz": bkg_dz, "dxdy": bkg_dxdy, \
    "$p_T$": bkg_pt, "$\eta$" : bkg_eta, "$\phi$": bkg_phi  })
    bkgJetsDF = pd.DataFrame({ "$p_T$": bkgJet_data[:,0], "$\eta$": bkgJet_data[:,1], \
    "$\phi$": bkgJet_data[:,2], "mass": bkgJet_data[:,3]  })

    #Extracting signal information
    sig_dz = signal_parts[:, :, 8].ravel()
    sig_dxdy = signal_parts[:, :, 9].ravel()
    #sig_dy = signal_parts[:, :,  10].ravel()
    sig_pt = signal_parts[:, :,  10].ravel()
    sig_eta = signal_parts[:, :, 11].ravel()
    sig_phi = signal_parts[:, :, 12].ravel()
    sigDF = pd.DataFrame({"dz": sig_dz, "dxdy": sig_dxdy, \
    "$p_T$": sig_pt, "$\eta$" : sig_eta, "$\phi$": sig_phi  })
    sigJetsDF = pd.DataFrame({ "$p_T$": signalJet_data[:,0], "$\eta$": signalJet_data[:,1], \
    "$\phi$": signalJet_data[:,2], "mass": signalJet_data[:,3]  })

    #plotting particle kinematics
    fig, axs = plt.subplots(3, 2, figsize=(12,12))
    #hep.cms.text("Phase 2 Simulation")
    #hep.cms.lumitext("PU 200 (14 TeV)")
    has_bkg = len(bkgDF) > 0
    for idx, ax in enumerate(axs.reshape(-1)[:-1]):
        bkgCol = bkgDF.columns[idx]
        sigCol = sigDF.columns[idx]
        bkgData = bkgDF[bkgCol]
        sigData = sigDF[sigCol]
        if has_bkg:
            xmin = min(np.min(bkgData), np.min(sigData))
            xmax = max(np.max(bkgData), np.max(sigData))
        else:
            xmin, xmax = np.min(sigData), np.max(sigData)
        #xmin = min(np.min(bkgData), np.min(sigData))
        #xmax = max(np.max(bkgData), np.max(sigData))

        ax.hist(sigData, bins=np.linspace(xmin, xmax, 51), alpha=0.8, label="signal", density=True)
        if has_bkg:
            ax.hist(bkgData, bins=np.linspace(xmin, xmax, 51), alpha=0.5, label="bkg", density=True)
        #ax.hist(bkgData, bins=np.linspace(xmin, xmax, 51), alpha=0.5, label="bkg", density=True)
        ax.set_xlabel(sigCol)
        ax.tick_params(axis="both", which="major")
        ax.set_yscale("log")
        ax.legend()

    plt.tight_layout()
    plt.show()
    if os.path.isdir(os.getcwd() + f"/{llpFolder}") == True:
        plt.savefig(os.getcwd() + f"/{llpFolder}/{method}_particle_kinematics.png")
        plt.savefig(os.getcwd() + f"/{llpFolder}/{method}_particle_kinematics.pdf")

    #plotting jet kinematics
    fig2, axs2 = plt.subplots(3, 1, figsize=(7,12))
    has_bkg_jets = len(bkgJetsDF) > 0
    for idx, ax2 in enumerate(axs2.reshape(-1)):
        bkgCol = bkgJetsDF.columns[idx]
        sigCol = sigJetsDF.columns[idx]
        bkgData = bkgJetsDF[bkgCol]
        sigData = sigJetsDF[sigCol]
        if has_bkg_jets:
            xmin = min(np.min(bkgData), np.min(sigData))
            xmax = max(np.max(bkgData), np.max(sigData))
        else:
            xmin, xmax = np.min(sigData), np.max(sigData)
        #xmin = min(np.min(bkgData), np.min(sigData))
        #xmax = max(np.max(bkgData), np.max(sigData))

        ax2.hist(sigData, bins=np.linspace(xmin, xmax, 51), alpha=0.8, label="signal", density=True)
        if has_bkg_jets:
            ax2.hist(bkgData, bins=np.linspace(xmin, xmax, 51), alpha=0.5, label="bkg", density=True)
        #ax2.hist(bkgData, bins=np.linspace(xmin, xmax, 51), alpha=0.5, label="bkg", density=True)
        ax2.set_xlabel(sigCol)
        ax2.tick_params(axis="both", which="major")
        ax2.legend()
        if sigCol == '$p_T$':
            ax2.set_yscale("log")
        else: continue
    plt.tight_layout()
    plt.show()
    if os.path.isdir(os.getcwd() + f"/{llpFolder}") == True:
        plt.savefig(os.getcwd() + f"/{llpFolder}/{method}_jet_kinematics.png")
        plt.savefig(os.getcwd() + f"/{llpFolder}/{method}_jet_kinematics.pdf")

