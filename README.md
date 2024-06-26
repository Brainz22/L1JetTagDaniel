# L1 LLP Tagger
Scripts and code used for the ongoing Level 1 LLP tagging project for CMS.


# Setting up Conda and cloning this repo: 
Conda will need to be set up in order to run any of the scripts, including launching jupyter notebooks. On the directory of your choice, run the following commands:
<pre>
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

export PATH="$HOME/miniconda3/bin:$PATH"
conda config --set auto_activate_base false

</pre>

After installing conda, we need to set up a virtual environment containing the modules used in the scripts. To create, and subsequently, activate the virtual environment, run:

<pre>
git clone /path/to/this/repo
conda env create -f environment.yml
conda activate L1JetTag
</pre>

Note inside the .yml file that the environment name will be L1JetTag.

# Reconstructing Jets:
The Ntuples containing particle data, i.e. events with particles, can be accessed using the grid through LXPLUS or LPC for instance. Once we can access the Ntuples, we need to extract the particle information and cluster them into jets with the `dataForgeScripts/dataForge.py` script. Such a file can be run using the following command with the corresponding arguments: 

`$ Python3 DataF.py </path/to/file> (using xrootd or another access mode) QCDpt30 30 50 0`

Order of the arguments is as follows:

path to file (using xrootd: `root://cmsxrootd.fnal.gov///store/...`)

tag = "QCDpt30" or "Stpt30" in this case. This is just a tag to be added to the name of the file.

ptCut = 30 (so, jets with Pt>30 GeV in this case).

trainPercent = 60 (60 % training data).

usePuppi = 0 (0 for pf, 1 for PUPPI).

# Removing Background Jets from Signal Sample: 
We have two main sources of background in our data: all jets reconstructed from the QCD Ntuples and some jets reconstructed from our signal Ntuples. The latter ones represent jets where the LLP is not matched to a jet in the event as determined by the DeltaR value. The script `removeBackground.py` can help us remove such jets to create a true-signal dataset. For that, we need to input the .h5 files (test and train) generated by dataForge.py from signal Ntuples. We can do that by running:

`python3 removeBackground.py "<path/train/filename.h5>" "<path/test/file.h5>"`

# Training Keras Model:

We can run can run this by passing three arguments the following way:
`python3 kerasModel.py "<SignalTrainFile.h5>" "<BackgroundTrainFile.h5>" "<JetData_TrainFile.h5>"`. These files should have resulted from the DataForge.py and/or `removeBackground.py`.
The `"<JetData_TrainFile.h5>"` will look something like `"sampleData...h5"` resulting from `DataForge.py'.

# ROC Curve from Keras Model:
Inside `ROC.py`, add paths of the testing data resulting from the DataForge.py and/or `removeBackground.py`.

# Training QKeras Model:
Inside of `qkerasModel.py`, add paths to the training files resulting from DataForge.py and/or `removeBackground.py`.

# ROC Curve from QKeras Model:
Inside `qkROC.py`, add paths of the testing data resulting from the DataForge.py and/or `removeBackground.py`.

# Conversion of ML models to HLS through HLS4ML:
Currently, the workflow is as follows:

 - Train and test the keras and qkeras models on LPC. From `kerasModel.py`, the file `L1JetTagModel.h5` will result. From the `qkerasModel.py`, the file `qkL1JetTagModel.h5` will result.
 - I copy these files into my PC using: `scp <lpc username>:</file/path/to/L1JetTagModel.h5> </directory/in/my/computer>`.
 
Then, I clone this repo inside Scully because we can source `Vivado hls` from there. Now, I have to copy the `/directory/in/my/computer/...Model.h5` files into the `L1JetTag` folder cloned in Scully.  Thus:
 - `scp </directory/in/my/computer/...Model.h5> <Scully username>:/path/to/L1JetTag/folder/>`

At this point, we are ready to convert the models contained in `.h5` files to the hls firmware using following the next steps:
 - `ssh -L localhost:8888:localhost:8888 <username>@scully.physics.ucsd.edu` (any 4 digits for local host.
 - `jupyter notebook --no-browser --port=8888 --ip localhost`

Now, open notebook `L1JetTagModel_hls_config.ipynb` or `qkL1JetTagModel_hls_config.ipynb`, respectively for keras and qkeras model. Then, inside of these file, make sure that the correct path is added for the `L1JetTagModel.h5` and `qkL1JetTagModel.h5` files. If the these files were copied to your `L1JetTag` folder in Scully, then you should not change anything, i.e.:
 - The line `model = load_model("L1JetTagModel.h5")` should stay like this in `L1JetTagModel_hls_config.ipynb`.
 - The line `qmodel = load_qmodel("qkL1JetTagModel.h5")` should stay like this in `qkL1JetTagModel_hls_config.ipynb`.

