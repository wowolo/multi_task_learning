#!/bin/bash

# check argument
if [ "$#" -ne 1 ]; then
    echo "no argument: tag missing"
	exit 0
fi

time=`date +%D_%T`
tmp_branch=seb_$1_$time

# update experiments files on git (make it temporary by defining a dedicated branch create delete)
git add ./experiments/* ./bash_files/*
git commit -m "update experiments for running '${1}' on Euler cluster"
git push origin master

# makes ssh connection and execute euler commands
cat ./bash_files/euler_commands.sh | ssh euler /bin/bash -s $1

# delete git branch and update 
git branch –delete tmp_branch