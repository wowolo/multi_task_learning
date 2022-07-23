*Updated: 22.7.22*
# 1. Repo Summary - Multi Task Learning 

This repository intends to provide a modular code framework to train neural networks on data. Its implementation hinges on the Python library [PyTorch](https://pytorch.org), an open source machine learning library, and [PyTorch Lightning](https://www.pytorchlightning.ai), a light wrapper for PyTorch code aiming to serve as a code style guide and to handle technical (hardware specific) aspects of training. 
Here, PyTorch Lightning is used within `core_code` module to define the core objects that have been defined with the intention to be based on the modular data and neural networks components to allow various degrees of freedom for customization. If these modules live up to their intentions, these can serve as a robust and unified code framework for a wide range of neural network experiments. 
Furthermore, the library allows the integration of the [Weights & Biases](https://wandb.ai/site) API which sorts and visualizes tracking data and is taylored to each experiment using callback. Additionally, the connection of the code with W&B allows the live supervision and hardware specific resource tracking your experiment via its website.

> Note that the predecessor to this repository can be found on github under https://github.com/wowolo/visualization and is kept as a historical reference.

$~$

**Table of contents:**

- [1. Repo Summary - Multi Task Learning](#1-repo-summary---multi-task-learning)
- [2. Repo Structure](#2-repo-structure)
- [3. Doing Experiments](#3-doing-experiments)
- [4. Software Setup](#4-software-setup)
  - [4.2. Python Packages](#42-python-packages)
  - [4.1. GitHub](#41-github)
  - [4.4. Weights & Biases](#44-weights--biases)
  - [4.3. Cluster (Euler)](#43-cluster-euler)
- [5. Making Changes](#5-making-changes)
  
$~$

# 2. Repo Structure

Sketch of the repository structure:

```
|--- bash_files
    |--- euler_commands.sh
    |--- remote_scp.sh
    |--- run_remote.sh
|--- core_code
    |--- model_lib
        |--- abcMLP.py
        |--- ... [other architectures]
        |--- util.py
    |--- util
        |--- config_extractions.py
        |--- default_config.py
        |--- function_lib.py
        |--- helpers.py
        |--- lightning.py
    |--- create_data.py
    |--- create_model.py
    |--- lightning_model.py
    |--- subREADME.md
|--- experiments
    |--- compositeSine
        |--- configs.py
        |--- logging_callback.py
        |--- manager.py
    |--- ... [other experiments]
    |--- subREADME.md
    |--- util.py
|--- main.py
|--- pre_main.py
|--- README.md
```

The core of the repository consists of the Python module `core_code` where modularity and universality are emphasized in order to possibly serve as (partial) code basis for future projects down the road. Within the `experiments` module we organize the experiment-specific configuration parameters and setup the respective experiments (i.e. in sketch above: compositeSine) based on the `core_code` module. 
The python file (pre_)main.py is then used to launch the program by calling the necessary objects which should have been prepared in the `experiments` module. Note that the pre_main.py file is part of the bash job-submission (on Euler) via ./bash_files/run_remote.sh. 
The structure of the repository should reflect and underline the underlying modular logic of the code and enforce conscious implementation decisions.

> More detailed explanations of the deeper repo structure can be found within the subREADME.md files of the respective sub-directories.
> File specific descptions can be found in the respective python files in the form of comments.

$~$

# 3. Doing Experiments
This section describes how one can launch `compositeSine` experiments using this repository and may help to illustrate the options and parametrizations provided by the code. To run the experiments refer to the next section, software setup, to determine which software setups may be needed for your experiments.
In general, each experiment should have a dedicated *Manager* Python class object, which is initialized with all the user specified configurations and starts the experiment(s) via its *run* method. Once the associated *Manager* is implemented in the `manager.py` file of the respective experiment, analogously to the `compositeSine` experiment, running the experiment is a simple exercise of adjusting the imports at the top of the (pre_)main.py files to load the experiment's configurations from its `configs.py` file and the respective *Manager* object. Furthermore, the code evaluates the given configurations consisting of several Python dictionaries in two steps. The first step allows the user to define several options for (possibly) each parameter and createa grid of configurations with all possible supplied configuration combinations. In the second step, each configuration of the resulting batch of experiments can then be supplied task specific configurations. 

> Refer to the `experiments/subREADME.md` file for more details and an example of this two step evaluation routine of the configurations.

$~$

The `compositeSine` experiment uses the following dictionaries to parametrize the experiment: 
- *configs_data* (data specific configurations for the *CreateData* object in `core_code/create_data.py`)
- *configs_architecture* (architecture specific configurations for the *CreateModel* object in `core_code/create_model.py`)
- *configs_training* (training specific configurations for the *LightningModel* object in `core_code/lightning_model.py`)
- *configs_custom* (custom configurations for the experiment's *Manager* object in `experiments/compositeSine/manager.py`)
- *configs_trainer* (*pytorch_lightning.Trainer* specific configurations for the *LightningModel.fit()* method in `core_code/lightning_model.py` which are directly passed to the *pytorch_lightning.Trainer* object - see [here](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) - for training the PyTorch Lightning model; supplied by bash arguments or default values in `core_code/util/default_config.py`)

All these dictionaries have hard coded keys (see `core_code/util/default_config.py` except for custom config see `experiments/compositeSine/manager.py`) which guarantees that all the dictionaries contain exactly these hard coded key by getting rid of any excessive, non-defined items in the input configurations and add possibly missing items based on the supplied default values defined in the respective Python files. The process of adding/deleting configuration items is specificallly described in Section 5.

> The possible configuration parameters are documented by the comments within the respective Python files (except *configs_custom*). 
  
$~$

Now, the user has the option to run the configuration induced batch of experiments either locally or the (ETH Euler) cluster (similar implementations should be also possible for other clusters). 
Running the experiments locally, the program can be started by the bash script `run_local.sh`. The *configs_trainer* are supplied by the array defined in the bash script and passed as input argument to the `main.py` program within the bash script.
Analogously, to run the experiments on the Euler cluste, use the `run_remote.sh` bash script.
Note that the underlying bash script `main_commands.sh` has to be customized before running either `run_local.sh` or `run_remote.sh`. Furthermore, the setup described in the next section Software Setup is needed to run the experiements without modifications to the code (i.e. Python environment, GitHub, Weights&Biases setup needed).



- wandb logging 
- ssssooooommmmmmeeeee

$~$

# 4. Software Setup

## 4.2. Python Packages

$~$

## 4.1. GitHub
init repo

$~$

## 4.4. Weights & Biases
.netrc

$~$

## 4.3. Cluster (Euler)
ssh, bash scripts, github repo on cluster (TODO implement updates by creating and deleting branches)

$~$

# 5. Making Changes