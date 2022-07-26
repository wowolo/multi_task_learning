#!/bin/bash

# example call from root directory:
# bash bash_files/run_remote.sh run_name

# check argument
if [ "$#" -ne 1 ]; then
    echo "no argument: tag missing"
	exit 0
fi

# CUSTOMIZE #
ssh_keyword=euler
#############

# update files on Euler based on local files
bash bash_files/remote_scp.sh

# makes ssh connection and execute main commands 
# TODO: add the name of tmp branch to access for experiments
cat ./bash_files/main_commands.sh | ssh $ssh_keyword /bin/bash -s $1 "remote" 