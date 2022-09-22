

<div align="center">

# Z-Stack-Enhancement
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/HelmholtzAI-Consultants-Munich/ML-Pipeline-Template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
<a href="https://github.com/pyscaffold/pyscaffoldext-dsproject"><img alt="Template" src="https://img.shields.io/badge/-Pyscaffold--Datascience-017F2F?style=flat&logo=github&labelColor=gray"></a>

</div>

# Description
This project is the next SOTA model in ML.

# Quickstart

## Create the project environment and install the zse package

### To use the PLI JSC Environment:

1. Make sure to deactivate all other virtual environments, e.g. run
```bash
conda deactivate
```

2. Setup and activate the included JSC environment: 
```bash
source environment/setup.sh
```

3. To activate the environment run
```bash
source environment/activate.sh
```

4. To deactivate the environment run
```
deactivate
```

5. To create a Jupyter kernel run
```bash
bash environment/create_kernel.sh
```

Checkout `environment/README.md` for more info (e.g. on how to create a Jupyter kernel).
`

### For every environment

Before using the template, one needs to install the project as a package:
```bash
pip install -e .
```


## Run the MNIST example
This pipeline comes with a toy example (MNIST dataset with a simple feedforward neural network). To run the training (resp. testing) pipeline, simply run:
```bash
python scripts/train.py
# or python scripts/test.py
```
Or, if you want to submit the training job to a submit cluster node via slurm, run:
```bash
sbatch scripts/train_juwels.sbatch
```
> * The experiments, evaluations, etc., are stored under the `logs` directory.
> * The default experiments tracking system is tensorboard. The `tensorboard` directory is contained in `logs`. To view a user friendly view of the experiments, run:
> ```bash
> # make sure you are inside logs (where mlruns is located)
> tensorboard --logdir logs/tensorboard/
> ```
> * To access the logs with tensorboard from the JSC filesystem you could either use [SSH tunneling](https://help.ubuntu.com/community/SSH/OpenSSH/PortForwarding) or as [sshfs](https://wiki.ubuntuusers.de/FUSE/sshfs/) mount
> * When evaluating (running `test.py`), make sure you give the correct checkpoint path in `configs/test.yaml`


## Versioneer

This project uses [Versioneer](https://github.com/python-versioneer/python-versioneer) to record package versions.

To create a new version use the [Git Tagging](https://git-scm.com/book/en/v2/Git-Basics-Tagging) utility:
```bash
git tag 1.2.3
```

To distribute it through gitlab push the tags and commits as
```bash
git push; git push --tags
``` 


# Project Organization
```
├── configs                              <- Hydra configuration files
│   ├── callbacks                               <- Callbacks configs
│   ├── datamodule                              <- Datamodule configs
│   ├── debug                                   <- Debugging configs
│   ├── experiment                              <- Experiment configs
│   ├── hparams_search                          <- Hyperparameter search configs
│   ├── local                                   <- Local configs
│   ├── log_dir                                 <- Logging directory configs
│   ├── logger                                  <- Logger configs
│   ├── model                                   <- Model configs
│   ├── trainer                                 <- Trainer configs
│   │
│   ├── test.yaml                               <- Main config for testing
│   └── train.yaml                              <- Main config for training
│
├── data                                 <- Project data
│   ├── processed                               <- Processed data
│   └── raw                                     <- Raw data
│
├── docs                                 <- Directory for Sphinx documentation in rst or md.
│
├── environment                          <- Computing environment
│   ├── requirements                            <- Python packages and JSC modules requirements
│   │
│   ├── activate.sh                             <- Activation script
│   ├── config.sh                               <- Environment configurations  
│   ├── create_kernel.sh                        <- Jupyter Kernel script
│   └── setup.sh                                <- Environment setup script
│
├── logs
│   ├── experiments                      <- Logs from experiments
│   ├── slurm                            <- Slurm outputs and errors
│   └── tensorboard/mlruns/...           <- Training monitoring logs
|
├── models                               <- Trained and serialized models, model predictions
|
├── notebooks                            <- Jupyter notebooks
|
├── reports                              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                                 <- Generated plots and figures for reports.
|
├── scripts                              <- Scripts used in project
│   ├── train_juwels.sbatch                     <- Submit job to slurm on JUWELS
│   ├── test.py                                 <- Run testing
│   └── train.py                                <- Run training
│
├── src/ml_pipeline_template             <- Source code
│   ├── datamodules                             <- Lightning datamodules
│   ├── models                                  <- Lightning models
│   ├── utils                                   <- Utility scripts
│   │
│   ├── testing_pipeline.py
│   └── training_pipeline.py
│
├── tests                                <- Tests of any kind
│   ├── helpers                                 <- A couple of testing utilities
│   ├── shell                                   <- Shell/command based tests
│   └── unit                                    <- Unit tests
│
├── .coveragerc                          <- Configuration for coverage reports of unit tests.
├── .gitignore                           <- List of files/folders ignored by git
├── .pre-commit-config.yaml              <- Configuration of pre-commit hooks for code formatting
├── setup.cfg                            <- Configuration of linters and pytest
├── LICENSE.txt                          <- License as chosen on the command-line.
├── pyproject.toml                       <- Build configuration. Don't change! Use `pip install -e .`
│                                           to install for development or to build `tox -e build`.
├── setup.cfg                            <- Declarative configuration of your project.
├── setup.py                             <- [DEPRECATED] Use `python setup.py develop` to install for
│                                           development or `python setup.py bdist_wheel` to build.
└── README.md
```
