#!/bin/bash

# check argument
if [ "$#" -ne 1 ]; then
    echo "no argument: tag missing"
	exit 0
fi

/bin/bash ./bash_files/main_commands.sh $1 "local"