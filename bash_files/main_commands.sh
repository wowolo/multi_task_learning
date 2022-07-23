#!/bin/bash

check argument
if [ "2" -gt "$#" ]; then
    echo "no argument: tag missing and run type missing"
	exit 0
fi
# get the job name (${1} is the tag provided as argument)
tag=${1}
run_type=${2}

################ General input ##################

# CUSTOMIZE START #
username=scheins # for Euler
project_path_local=/Users/sebastianschein/dev/master_thesis/multi_task_learning
project_path_remote=/cluster/home/$username/master_thesis/multi_task_learning
python_env=mtl_env
###################

# set trainer configurations 
config_trainer=(
    accelerator:auto
    strategy:ddp_find_unused_parameters_false
    devices:auto
    auto_select_gpus:False
    deterministic:False
    # default_root_dir
    # auto_lr_find:False
    # amp_backend
    fast_dev_run:False
    # precision
    enable_progress_bar:False
    max_epochs:1
    max_time:00:09:00:00 # 00:12:00:00 - 12 hours
)

#################################################

if [ $run_type == "local" ]; then

    $project_path_local/$python_env/bin/python $project_path_local/main.py --experimentbatch_name $tag --config_trainer ${config_trainer[@]}

elif [ $run_type == "remote" ]; then

    echo "#######################################################"
    echo "start"
    echo "#######################################################"

    tmp_branch=${3}

    # go to main directory and update via github
    cd $project_path_remote
    git stash
    git stash clear
    git pull origin master
    git checkout $tmp_branch

    # ressource allocation
    max_time="09:15" # maximum time (hour:second") allocated for the job (max 120:00 / large value implies low priority)
    n_core="1" # number of core (large value implies low priority)
    memory="46000" # memory allocation (in MB) per core (large value implies low priority)
    scratch="0" # disk space (in MB) for temporary data per core
    n_gpus=8

    # get the log filename
    log="${tag}_out.txt"
    err="${tag}_error.txt"

    # load python environment specified by second input
    module load gcc/8.2.0 python_gpu/3.9.9 eth_proxy
    source $project_path_remote/$python_env/bin/activate

    # compute the number of configs that will be handled separately by a job array
    python $project_path_remote/pre_main.py
    num_configs=$?
    tag=$tag'[1-'$num_configs']'

    # submit the job
    bsub -G ls_math -J $tag -o $log -e $err -n $n_core -W $max_time -N -R "rusage[mem=$memory, ngpus_excl_p=$n_gpus]" "python $project_path_remote/main.py --num_config \$LSB_JOBINDEX --experimentbatch_name $tag --config_trainer ${config_trainer[@]}"

    # display the current queue
    bbjobs

    git checkout master
    git branch -D $tmp_branch


    echo "#######################################################"
    echo "end"
    echo "#######################################################"


else
    echo "The run type has neither been 'local' nor 'remote' and is therefore not well defined."
fi

exit