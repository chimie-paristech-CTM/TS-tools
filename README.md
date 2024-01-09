# TS-tools

This is the repository corresponding to the TS-tools project.

### Setting up the environment

To set up the ts_tools conda environment:

```
conda env create -f environment.yml
```

To install the TS-tools package, activate the ts-tools environment and run the following command within the TS-tools directory:

```
pip install .
```

Additionally, Gaussian16 needs to be available. In HPC environments, this can typically be achieved by loading the corresponding module to the path:

```
module load gaussian/g16_C01
```

### Generating TS guesses at xTB level-of-theory

TS guesses at xTB level of theory can be generated in parallel by running the following command:

```
python run_scripts/run_ts_searcher.py [--input-file data/reactions_am.txt] [--xtb-external-path "xtb_external_path/xtb_external.py"]
```

where the ’input-file’ command line option corresponds to the location of the .txt-file containing the reaction SMILES (the default value ’reactions_am.txt’ corresponds to the benchmarking reactions, 
and  the ’xtb-external-path’ option corresponds to the location of the script to use xTB as an external method in Gaussian16 (copied from the Jensen group's [xtb_gaussian](https://github.com/jensengroup/xtb_gaussian/blob/main/xtb_external.py) repository).  

Additional options can also be provided:

1. '--reactive-complex-factors-intra': Specifies a list of floating-point numbers representing reactive complex factors for intra-molecular interactions.
2. '--reactive-complex-factors-inter': Specifies a list of floating-point numbers representing reactive complex factors for inter-molecular interactions.
3. '--solvent': Specifies the name of the solvent (needs to be supported both in xTB and Gaussian16, e.g., 'water')
4. '--freq-cut-off': Specifies the imaginary frequency cut-off used during filtering of plausible starting points for transition state guess optimization.
5. '--target-dir': Specifies the working directory in which all files will be saved; final reactant, product and TS guess geometries (as well as the TS .log file) are saved in another directory with the ’final_’ prefix.

### Validating TS guesses at DFT level of theory  
