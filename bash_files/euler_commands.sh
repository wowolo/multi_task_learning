#!/bin/bash
#
# main bash script for submitting a python/EULER job:
#     - allocate the resources
#     - load the python environment
#     - submit the job
#
# to run the script:
#     - execute "chmod +x python_job.sh"
#     - execute "./python_job.sh
#     - the argument "tag" is passed to the MATLAB code
#
############################################################################

check argument
if [ "$#" -ne 1 ]; then
    echo "no argument: tag missing"
	exit 0
fi
# get the job name (${1} is the tag provided as argument)
tag="${1}"

username=scheins
project_path=/cluster/home/$username/master_thesis/visualization
python_env=visual_env

# go to main directory and update via github
cd $project_path
git stash
git pull origin master

echo "#######################################################"
echo "start"
echo "#######################################################"

# ressource allocation
max_time="09:15" # maximum time (hour:second") allocated for the job (max 120:00 / large value implies low priority)
n_core="1" # number of core (large value implies low priority)
memory="46000" # memory allocation (in MB) per core (large value implies low priority)
scratch="0" # disk space (in MB) for temporary data per core
n_gpus=8

# set trainer configurations - partially implied by ressource allocation (do not decouple)
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
    max_epochs:2048
    max_time:00:09:00:00 # 00:12:00:00 - 12 hours
)

# get the log filename
log="${tag}_out.txt"
err="${tag}_error.txt"

# load python environment specified by second input
module load gcc/8.2.0 python_gpu/3.9.9 eth_proxy
source $project_path/$python_env/bin/activate

# compute the number of configs that will be handled separately by a job array
python $project_path/pre_main.py
num_configs=$?
tag=$tag'[1-'$num_configs']'

# submit the job
bsub -G ls_math -J $tag -o $log -e $err -n $n_core -W $max_time -N -R "rusage[mem=$memory, ngpus_excl_p=$n_gpus]" "python $project_path/main.py --num_config \$LSB_JOBINDEX --experimentbatch_name $tag --config_trainer ${config_trainer[@]}"

# display the current queue
bbjobs


echo "#######################################################"
echo "end"
echo "#######################################################"

exit