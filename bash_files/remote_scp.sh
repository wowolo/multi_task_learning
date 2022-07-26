#!/bin/bash

# CUSTOMIZE #
username=scheins
remote_dir=/cluster/home/$username/master_thesis/multi_task_learning/
local_dir=~/dev/master_thesis/multi_task_learning/
ssh_idrsa=~/.ssh/id_ed25519_euler
#############

# bash files
scp -i $ssh_idrsa -r $local_dir/bash_files $username@euler.ethz.ch:$remote_dir

# core_code module
scp -i $ssh_idrsa -r $local_dir/core_code $username@euler.ethz.ch:$remote_dir

# experiments module
scp -i $ssh_idrsa -r $local_dir/experiments $username@euler.ethz.ch:$remote_dir

# pre_main.py script
scp -i $ssh_idrsa -r $local_dir/pre_main.py $username@euler.ethz.ch:$remote_dir

# main.py script
scp -i $ssh_idrsa -r $local_dir/main.py $username@euler.ethz.ch:$remote_dir