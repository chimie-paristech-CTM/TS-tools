# TS-tools

This is the repository corresponding to the TS-tools project. 

### Setting up the environment

To set up the ts_tools conda environment:

```
conda env create -f environment.yml
```

To install the TS-tools package, activate the ts_tools environment and run the following command within the TS-tools directory:

```
pip install .
```

Additionally, Gaussian16 needs to be available. In HPC environments, this can typically be achieved by loading the corresponding module to the path:

```
module load gaussian/g16_C01
```

### 
