*Updated: 24.7.22*
# 1. Repo Summary - Multi Task Learning 

This repository intends to provide a modular code framework to train neural networks on data. Its implementation hinges on the Python library [PyTorch](https://pytorch.org), an open source machine learning library, and [PyTorch Lightning](https://www.pytorchlightning.ai), a light wrapper for PyTorch code aiming to serve as a code style guide and to handle technical (hardware specific) aspects of training. 
Here, PyTorch Lightning is used within `core_code` module to define the core objects that have been defined with the intention to be based on the modular data and neural networks components to allow various degrees of freedom for customization. If these modules live up to their intentions, these can serve as a robust and unified code framework for a wide range of neural network experiments. \
The explicit goal of supporting multi-task learning problems is handled by task specific designations of configurations, data, data loaders and models. Formally, this is supported by Python dictionaries ('task dictionaries' with 'task_0',...,'task_n' as keys for n+1 tasks) throughout the different stages in the code. \
Furthermore, the library allows the integration of the [Weights & Biases](https://wandb.ai/site) API which sorts and visualizes tracking data and is taylored to each experiment using callback. Additionally, the connection of the code with W&B allows the live supervision of the network's training and the hardware specific resource tracking of your experiment via its website.

> Note that the predecessor to this repository can be found on github under https://github.com/wowolo/visualization and is kept as a historical reference.

$~$

**Table of contents:**

- [1. Repo Summary - Multi Task Learning](#1-repo-summary---multi-task-learning)
- [2. Repo Structure](#2-repo-structure)
- [3. Doing Experiments](#3-doing-experiments)
- [4. Software Setup](#4-software-setup)
  - [4.1. GitHub](#41-github)
  - [4.2. Python Packages](#42-python-packages)
  - [4.3. Weights & Biases](#43-weights--biases)
  - [4.4. Cluster (Euler)](#44-cluster-euler)
- [5. Making Changes](#5-making-changes)
  - [5.1. New Experiment (some guideline ideas)](#51-new-experiment-some-guideline-ideas)
  - [5.2. New Configs Items / Change Default Configs](#52-new-configs-items--change-default-configs)
  - [5.3. Data](#53-data)
  - [5.4. Model(s)](#54-models)
  - [5.5. Batching (batch creation in training)](#55-batching-batch-creation-in-training)
  - [5.6. Logging (via W&B)](#56-logging-via-wb)
  
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

The core of the repository consists of the Python module `core_code` where modularity and universality are emphasized in order to possibly serve as (partial) code basis for future projects down the road. Within the `experiments` module we organize the experiment-specific configuration parameters and set up the respective experiments (i.e. in sketch above: compositeSine) based on the `core_code` module. \
The python file (pre_)main.py is then used to launch the program by calling the necessary objects which should have been prepared in the `experiments` module. Note that the pre_main.py file is part of the bash job-submission (on Euler) via ./bash_files/run_remote.sh. \
The structure of the repository should reflect and underline the underlying modular logic of the code and enforce conscious implementation decisions.

> More detailed explanations of the deeper repo structure can be found within the subREADME.md files of the respective sub-directories.
> File specific descptions can be found in the respective python files in the form of comments.

$~$

# 3. Doing Experiments
This section describes how one can launch `compositeSine` experiments using this repository and may help to illustrate the options and parametrizations provided by the code. To run the experiments refer to the next section, software setup, to determine which software setups may be needed for your experiments. \
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

Now, the user has the option to run the configuration induced batch of experiments either locally or the (ETH Euler) cluster (similar implementations should be also possible for other clusters). \
Running the experiments locally, the program can be started by the bash script `run_local.sh`. The *configs_trainer* are supplied by the array defined in the bash script and passed as input argument to the `main.py` program within the bash script. \
Analogously, to run the experiments on the Euler cluste, use the `run_remote.sh` bash script. \
Note that the underlying bash script `main_commands.sh` (section commented for customization) has to be customized before running either `run_local.sh` or `run_remote.sh`. Furthermore, the setup described in the next section Software Setup is needed to run the experiements without modifications to the code (i.e. Python environment, GitHub, Weights&Biases setup needed).

$~$

# 4. Software Setup
This sections describes the preparations that should be made software wise to use the complete range of implementations without potential modification to the code.

$~$

## 4.1. GitHub
It is adviced to copy this remote repository to ones local repository using git as well as creating a remote repository accesible from the cluster. Specifically, the version tracking will be used to make changes to the code locally and launch the updated code via the local bash script (using temporary branches in the case of temporary changes to the configurations, avoiding unnecessary commits in the master branch). \
([Reference](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) for your own repository.) 

$~$

## 4.2. Python Packages
The Python virtual environment should be generated from the `requirements.txt` file (see [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)) to guarantee that the Python packages fulfill the requirements and be located directly in the project's root directory.

$~$

## 4.3. Weights & Biases
Apart from the necessary Python packages leveraging the W&B API, which we have covered by the previous step, you have to login to your (existing) W&B account. See [here (1. Set up wandb)](https://docs.wandb.ai/quickstart) for reference.

> Note that this has to be done for your account on Euler as well if you want to launch jobs on the cluster.
> Tip: If you encounter probelems with the automatic W&B login, referenced above, one should check in ones .netrc if the necessary login details for W&B are recorded in this file.

$~$

## 4.4. Cluster (Euler)
There are two steps to be taken to set up everything around Euler in order to be able launch jobs on Euler locally via the local bash script `bash_files/run_remote.sh`. \
The first one is to guarantee that a git repository is set up on your Euler account to synchronize changes made to the code locally and create the the Python virtual environment from `requirements.txt`. See Section 4.1 regarding GitHub and [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) regarding the virtual environment. \
The second one is to configure the SSH keys to login to your account without having to repeatedly type the password. Follow the instructions on *SSH keys* by the Euler Cluster Support [here](https://scicomp.ethz.ch/wiki/Accessing_the_clusters). \
Lastly, customize your bash_files. For the ssh_keyword in `bash_files/run_remote.sh` set it such that you would login by
```
# in shell
ssh_keyword=... # your “ssh keyword”
ssh $ssh_keyword 
```

$~$

# 5. Making Changes
The code objects within this multi taks learning repository haven been designed with the goal to provide the necessary modularity and degrees of freedom to accomodate various use-cases and potential modifications down the road. We will run through the scenarios which have motivated the code architecture. \
Moreover, we hope, that in case of a scenario which is not covered in by one of the following sub-sections, the logic and principles of the existing code structure provides a suitable basis to integrate such new scenarios.

$~$

Ultimately, the goal should be to define the *LightningModel* in `core_code/lightning_model.py` using a (task-specific) PyTorch Lightning *DataModule* from `core_code/util/lightning.py` and (task-specific) PyTorch model(s). This means that any (potentially re-factored/customized) experiment which has these previously mentioned objects can be encapsulated in this repository's multi-task code architecture, providing a uniform structure, the ability to run the code remotely on (mutliple) CPUs or GPUs and log the experiment via W&B.

$~$

## 5.1. New Experiment (some guideline ideas)
The current idea for creating new experiments is to create a new directory in the exisiting `experiments/` directory and define at its core the respective *Manager* class which inherits the methods to create parameter grids based on the given configurations in `experiments/[new_experiment]/configs.py` from its parent class *BasicManager* in `experiments/util.py`. A *configs_custom* dictionary might help to parametrize certain aspects of the manager itself (as done for the *Manager* of the compositeSine experiment).
Furthermore, the *LoggingCallback* class provides callbacks into the training of the neural network using its parent class *pytorch_lightning.Callback*. These allow to cache metrics/losses/etc. (via state attributes, see `experiments/compositeSine/logging_callback.py`) and also log them via W&B (refer to Section 5.6 for more details).

$~$

## 5.2. New Configs Items / Change Default Configs
To add new configuration items in the underlying Python dictionaries one has to access the functions in `core_code/util/default_config.py`. There one can also adjust the default values. Note that *configs_custom* however is part of the *Manager* class in `experiments/compositeSine/manager.py`. 

> The functions for initializing/checking the user configurations are named init_config_* and make use of the _make_init_config helper function found in `core_code/util/default_config.py`.

$~$

## 5.3. Data
TO UPDATE!
The *CreateData* class in `core_code/create_data.py` creates (noisy) data for a given function. If the intended data cannot be modelled based on such a mechanism or if there exists an already given dataset, the user has the option to introduce its own data. For the training via the PyTorch Lightning *Trainer* the data has to be loaded into a PyTorch *Dataset* and then loaded into a PyTorch *Dataloader* - see `core_code/util/lightning.py` for existing implementations of these two objects. By conforming to the data layout '(x, y, task_activity)' the new data can be embedded into the existing code layout.

$~$

## 5.4. Model(s)
TO UPDATE!
Analogously to the previous point regarding data, the user can also introduce their own (task-specific) model(s). These can be any PyTorch model(s) with the only limitation being that these have to align with the input and ouptut dimensions induced by the (task-specific) data.

$~$

## 5.5. Batching (batch creation in training)
The creation of the batches throughout the training of the neural network is dependent on the *DataLoaders* class in `core_code/util/lightning.py` and supplies the data *DataModule* in `core_code/util/lightning.py` and ultimately the LightningModel in `core_code/lightning_model.py` with a task dictionary of train, validation and test dataloaders.


$~$

## 5.6. Logging (via W&B)
In this implemetation we have used the *pytorch_lightning.logger.WandbLogger* to log metrics and plots after setting up the W&B API as outlined in Section 4.3. The logging can be highly customized - our choice to implement this was to create the *LoggingCallback* class to have the best control possible of the logging throughout the training and we can recommend this way of logging for any other experiments due to its flexibility, its uniformity in (code) structure and its modularity. A working implementation can be found in `experiments/compositeSine/logging_callback.py`.