*Updated: 26.7.22*
# 1. Repo Summary - Multi Task Learning 

This repository intends to provide a modular code framework to train (multi-task) neural networks on data. Its implementation hinges on the Python library [PyTorch](https://pytorch.org), an open source machine learning library, and [PyTorch Lightning](https://www.pytorchlightning.ai), a light wrapper for PyTorch code which serves as a code style guide and handles the technical (hardware specific) aspects of CPU/GPU training. \
Here, PyTorch Lightning is used within the `core_code` module to define the core objects which modularize the data and neural network model components. These modules provide a building block wise construction of the experiments with ample opportunities of customization (by using the parameters, callbacks or by embedding ones own objects). This can serve as a robust and unified framework for a wide range of (multi-task) neural network experiments. 

$~$

The explicit goal of supporting multi-task learning problems is handled by task specific identification of the configurations, data and models. \
Furthermore, the library allows the integration of the [Weights & Biases](https://wandb.ai/site) API which stores and visualizes tracking data and can be taylored to each experiment using appropriate callbacks. Moreover, W&B allows live supervision of the network's training and hardware specific resource tracking of via its website (account needed).

$~$

> Note that the predecessor to this repository can be found on GitHub under https://github.com/wowolo/visualization and is kept as a historical reference.

$~$

**Table of contents:**

- [1. Repo Summary - Multi Task Learning](#1-repo-summary---multi-task-learning)
- [2. Repo Structure](#2-repo-structure)
- [3. Doing Experiments](#3-doing-experiments)
- [4. Software Setup](#4-software-setup)
  - [4.1. GitHub](#41-github)
  - [4.2. Python Virtual Environment](#42-python-virtual-environment)
  - [4.3. Weights & Biases](#43-weights--biases)
  - [4.4. Cluster (Euler)](#44-cluster-euler)
- [5. Making Changes](#5-making-changes)
  - [5.1. New Experiment (some guideline ideas)](#51-new-experiment-some-guideline-ideas)
  - [5.2. Change Config Items and their Default Values](#52-change-config-items-and-their-default-values)
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
    |--- lightning_multitask.py
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

$~$

The core of the repository consists of the appropriately named Python module `core_code` where modularity and universality have been emphasized in its development in order to serve as (partial) code basis for future projects down the road. Within the `experiments` module we organize the experiment-specific configuration parameters and set up the respective experiments via a *Manager* class (i.e., in sketch above: compositeSine) where we use the respective data, model and training objects from the `core_code` module. \
The python files (pre_)main.py are then used to launch the program by calling the *Manager* object defined in the associated experiment directory (such as `compositeSine`) within the `experiments` module. \
The shell scripts in `bash_files` can be used to launch the code from the command line for local and remote runs. Note that the pre_main.py file is part of the bash job-submission (on Euler) via `./bash_files/run_remote.sh`. \
The structure of the repository should reflect and clarify the underlying modular logic of the code and enforce conscious implementation decisions.

$~$

> Detailed descriptions of the objects can be found in the respective Python scripts in the form of the objects' documentations. 

$~$

# 3. Doing Experiments

Using the example of the `compositeSine` experiment, this section describes how one can launch experiments using this repository. Before you actually run the code you should refer to the next section, software setup, to make the necessary preparations for your experiments. \
In general, each experiment should have a dedicated *Manager* Python class object in `manager.py` , which is initialized with all the user specified configurations for the specific experriment and starts the run(s) via its *run* method. \
Furthermore, the code evaluates the given configurations consisting of several Python dictionaries in two stages. In the first stage every value in the configuration is put into a list if it is not already one. Then a parameter grid is created by going through all possible combinations of values for the distinct configuration keys. In the second stage each resulting configuration dictionary, which is just one point from the previously created parameter grid, is iterated over. The respective values of one such dictionary are checked for the following question: Is it a task-specific value or is it a value which is valid for all tasks? A task-specific value is given as a dictionary with task keys (of the format 'task_{i}' - i-th task) and the associated values. Correspondingly, any value which is not of type dictionary is evaluated as non task-specific value, i.e., valid for all tasks.

$~$

> Refer to the `experiments/compositeSine/configs.py` scipt for an example of the resulting format of such configurations.

$~$

The `compositeSine` experiment uses the following dictionaries to parametrize the experiment: 
- *configs_data* (data specific configurations for the *CreateData* object in `core_code/create_data.py`)
- *configs_architecture* (architecture specific configurations for the *CreateModel* object in `core_code/create_model.py`)
- *configs_training* (training specific configurations for the *LightningMultitask* object in `core_code/lightning_multitask.py`)
- *configs_custom* (custom configurations for the experiment's *Manager* object in `experiments/compositeSine/manager.py`)
- *configs_trainer* (*pytorch_lightning.Trainer* specific configurations for the *LightningModel.fit()* method in `core_code/lightning_model.py` which are directly passed to the *pytorch_lightning.Trainer* object - see [here](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) - for training the PyTorch Lightning model; the configurations are supplied by bash arguments or default values in `core_code/util/default_config.py`)
  
$~$

All these dictionaries have fixed keys (set in `core_code/util/default_config.py` except for custom config see `experiments/compositeSine/manager.py`) which guarantees that all the dictionaries contain exactly the necessary keys by getting rid of any excessive, non-defined items in the input configurations and add possibly missing items based on the supplied default values defined in the respective Python files. The process of adding new or deleting old configuration items is specificallly described in Section 5.
  
$~$

The runs can either be made locally or remotely on the Euler cluster. \
Running the experiments locally, the program can be started by the bash script `bash_files/run_local.sh`. The *configs_trainer* is supplied by the array defined in the shell script `bash_files/euler_commands.sh` and passed as input argument to the `main.py` program within the shell script. \
Analogously, to run the experiments on the Euler cluster, use the `bash_files/run_remote.sh` bash script. \
Note that **all** the underlying bash scripts in `bash_files` have to be customized before running either application, `run_local.sh` or `run_remote.sh`. (These section are marked by a *CUSTOMZE* comment environment in each shell shell script.)

$~$

# 4. Software Setup
This sections describes the software setup in which the code has been developed and all applications work without modifications. However, feel free to make any adjustments that you feel might work for you!

$~$

## 4.1. GitHub

The first step would be to clone the repository on your own device. \
(See the GitHub [instructions](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).) 

$~$

## 4.2. Python Virtual Environment

The Python virtual environment should be created directly in the root directory. Then, after activating the environment you can use *pip* and the `requirements.txt` file to download all the right packages and the right versions. See [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)) for a complete instruction on all the necessary steps.

$~$

## 4.3. Weights & Biases

Apart from the Python W&B API, which we is already included in your virtual environment after the previous step, you also have to login to your (existing) W&B account. See [here (1. Set up wandb)](https://docs.wandb.ai/quickstart) for the complete reference in order to automatically connect to your W&B account on each session to your device.

$~$

> Note that this has to be done for your account on Euler as well if you want to launch jobs on the cluster. 

> Personal tip: If you encounter probelems with the automatic W&B login, referenced above, one can check in the .netrc (user) directory if the necessary login details for W&B are recorded in this file. If not a quick google search might and help you in this step.

$~$

## 4.4. Cluster (Euler)

There are two steps to be taken to set up everything around Euler in order to be able launch jobs on Euler locally via the local bash script `bash_files/run_remote.sh`. \
The first one would be to create the virtual environment in your remote root directory as well by following the steps in Section 4.2. \
The second step is to configure the SSH keys to login to your account without having to repeatedly type the password. Follow the instructions on *SSH keys* by the Euler Cluster Support [here](https://scicomp.ethz.ch/wiki/Accessing_the_clusters). Lastly, when customizing your bash_files set the ssh_keyword in `bash_files/run_remote.sh`, i.e., you would login by
```
# in shell
ssh_keyword=... # your “ssh keyword”
ssh $ssh_keyword 
```

$~$

> You need to customize all shell scripts in the bash_files to use the application to run it remotely and locally.

$~$

# 5. Making Changes

The objects within the `core_code` module have been designed with the goal to allow the user as much modification possibilities as possible. In general, lighter modifications would entail the use of existion parametrizations (and potentially adding new ones) while heavier modifications might result in the complete replacement of `core_code` object by custom ones. \
Moreover, while we expect that customizations might happen quite frequently, we hope that the code architecture implied by the `core_code` objects create a general framework for all these multi-task neural network experiments and provide a uniform style guide for similar experiments.

$~$

Ultimately, the 'Python goal' of all these experiments should be to define the *LightningMultitask* object in `core_code/lightning_multitask.py` using a PyTorch Lightning *DataModule* from `core_code/util/lightning.py` and (potentially task-specific) PyTorch model(s). This means that any experiment which can be build upon these objects can also be encapsulated in this repository's multi-task code architecture. And hence results in a uniform structure and can be neatly wrapped into the PyTorch Lightning package and connected to the W&B API with limited added effort and a high degree of customization.

$~$

## 5.1. New Experiment (some guideline ideas)

One idea for creating a new experiments woudl be to take the following steps: \ 
Create a new directory called `[new_experiment]` in the exisiting `experiments/` directory and define at its core the *Manager* class in the Python script called `experiments/[new_experiment]/manager.py`. From its parent class *BasicManager* in `experiments/util.py` the class inherits the methods necessary to create parameter grids based on the given configurations. \
Provide the dictionary configurations in a separate sript called `experiments/[new_experiment]/configs.py`. A *configs_custom* dictionary might help to parametrize certain aspects of the manager itself (as done for the *Manager* of the compositeSine experiment). \
Furthermore, implement the *LoggingCallback* class in `experiments/[new_experiment]/logging_callback.py` to create callbacks into the PyTorch Lightning training routine of the neural network by using *pytorch_lightning.Callback* as its parent class. You can log any quantities and media but also cache data such as metrics/losses/etc. (via state attributes, see `experiments/compositeSine/logging_callback.py`) for use later on in the training routine. These are logged on W&B (refer to Section 5.6 for more details).

$~$

## 5.2. Change Config Items and their Default Values

To change the viable configuration items in the underlying Python dictionaries one has to access the functions in the script `core_code/util/default_config.py`. There one can also adjust the default values. Note that *configs_custom* however is part of the *Manager* class in `experiments/compositeSine/manager.py`. \
Furthermore, note that for certain config extractions defined in `core_code/util/config_extraction.py` it is possible to provide the value 'custom' and an additional callback that “inserts the relevant function directly”. Refer to the source code in these scripts for the concrete details and options.

$~$

> The functions for initializing/checking the user configurations are named init_config_* and make use of the _make_init_config helper function found in `core_code/util/default_config.py`.

$~$

## 5.3. Data

The *CreateData* class in `core_code/create_data.py` provides training, validation and test data for a given function. Any functions can be simply added to the function library in `core_code/util/function_lib.py` and selected via the configurations by the 'f_true' key. \
If the intended data cannot be modelled by such a mechanism or if want to work with an already given dataset, we have the option to introduce the data by creating our own *DataModule* as in `core_code/lightning_multitask.py`. The first step would be to separate the data into the format (x, y, task_activity) in order to be passed into the *Dataloaders* function in the `core_code/util/lightning.py` script. Next the 'batching_strategy' within the *Dataloaders* function in the same script has to be set. For customization choose among existing strategies, add new batching strategies to the source code directly or set 'batching_strategy' to 'custom' and provide a custom batching strategy via the configuration item with key 'batching_strategy_callback'. Refer to the source code and its documentation for format constraints. In the last step the *DataModule* can be defined with the training, validation (and test) data and the previously defined dataloaders.

$~$

## 5.4. Model(s)

Concerning models we can use either one uniform PyTorch model (implying that the tasks need to have identical input and output dimensions) or we can create (task-specific) model(s) which might have intersecting parameters (where input and output dimensions can differ from task to task). Refer to the source code for the *LightningMultitask* class in the `core_code/lightning_multitask.py` script for details. \
To summarize, we can provide any (task-specific) PyTorch model(s) with the only limitation being that these have to align with the input and ouptut dimensions induced of the associated (task-specific) data. Already implemented architecture can be easily retrieved via the *CreateModel* function in the `core_code/create_model.py` script (where the underlying models are stored in the `core_code/model_lib` directory).

$~$

## 5.5. Batching (batch creation in training)

The creation of the batches throughout the training of the neural network is dependent on the *Dataloaders* function in `core_code/util/lightning.py` and supplies the data *DataModule* in `core_code/util/lightning.py` with a task dictionary of train, validation and test dataloaders. These then supply task specific data for the training, validation and test steps respectively in the form of task dictionaries. Find more details regarding the dataloaders in the *Dataloaders* documentation, regarding the training/validation/test steps in the source code of the *LightningMultitask* class in the `core_code/lightning_multitask.py` and also in the general [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/en/stable/guides/data.html).

$~$

## 5.6. Logging (via W&B)

In the any `[new_experiment]` implementation one can use the object *pytorch_lightning.logger.WandbLogger* as logger to log metrics and medias. The logging can be made experiment agnostic as well as it can be highly customized - this is made possible by the *LoggingCallback* class in the `experiments/[new_experiment]/logging_callback.py` script with the PyTorch Lightning parent class *Callback*. The callbacks allow us access to various points throughout the training, validation and test steps - see the [official PyTorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html) for more. \
We recommend this way of logging due to its high modularity, flexibility and uniform code structure. A working implementation can be found in `experiments/[new_experiment]/logging_callback.py`.

$~$

> Recall that the steps in Section 4.3 regarding setting up W&B are necessary in order to make the W&B logging work via with access via the website and its interface.