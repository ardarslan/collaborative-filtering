# Computational Intelligence Lab 2021

Project: Collaborative Filtering

Group name on Kaggle: Bayesians

## Reproducing results


### 1. Install conda if it's not already installed

Download installer from here: https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh


Give execution permission to the installer:

    chmod +x Miniconda3-latest-Linux-x86_64.sh

Run the installer:

    ./Miniconda3-latest-Linux-x86_64.sh

Open a new terminal and verify that conda is installed.


### 2. Create the environment

    conda env create -f environment.yml


### 3. Activate the environment

    conda activate cil_project


### 4. Make the configurations in main.py

    model should be "bayesian_gc_svdpp" or "gc_svdpp". Use "gc_svdpp" for Non-Bayesian SVD++ model which uses Graph Convolution Networks. Use "bayesian_gc_svdpp" for the Bayesian model which also calculates parameter and prediction uncertainties.

    device should be "cuda" or "cpu". Use "cuda" if you will submit the job to a GPU.

    data_path should be the path where the data is located.

    save_dir should be the path where the predictions will be saved to.


### 5. Submit the training task to GPU with the following commands (indicated time is necessary to reproduce results)

    cd src/

    bsub -n 4 -W 24:00 -o logs -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python main.py

Predictions and uncertainty plots (if Bayesian mode is active) will be saved under the directory "save_dir".

**Please refer to the README in src/baselines to reproduce the baseline results.**

Some of this code was adapted from https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcmc


