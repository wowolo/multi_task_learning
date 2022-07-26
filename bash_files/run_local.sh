#!/bin/bash

# example call from root directory:
# bash bash_files/run_local.sh run_name

# check argument
if [ "$#" -ne 1 ]; then
    echo "no argument: tag missing"
	exit 0
fi

/bin/bash ./bash_files/main_commands.sh $1 "local"