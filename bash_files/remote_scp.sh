#!/bin/bash
username=scheins
remote_dir=/cluster/home/$username/master_thesis/visualization/experiments/compositeSine/storage/*
local_dir=~/dev/master_thesis/visualization/experiments/compositeSine/storage/
ssh_idrsa=~/.ssh/id_ed25519_euler

scp -i $ssh_idrsa -r $username@euler.ethz.ch:$remote_dir $local_dir