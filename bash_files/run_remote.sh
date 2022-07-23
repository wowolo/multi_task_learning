#!/bin/bash

# check argument
if [ "$#" -ne 1 ]; then
    echo "no argument: tag missing"
	exit 0
fi

time=`date +%D_%T`
tmp_stash=user_$time
tmp_branch=user_$1_$time

# update experiments files on git (make it temporary by defining a dedicated branch create delete)

git stash push -m ${tmp_stash} 
git checkout -b $tmp_branch
git stash apply stash^{/${tmp_stash}}
git add ./experiments/* ./bash_files/*
git commit -m "temporary commit on temporary branch for experiments '${1}' on Euler cluster"
git push origin master

# makes ssh connection and execute main commands 
# TODO: add the name of tmp branch to access for experiments
cat ./bash_files/main_commands.sh | ssh euler /bin/bash -s $1 "remote" $tmp_branch

# delete git branch, retrieve stashed changes and delete stash
git checkout master
git branch -D $tmp_branch
git stash apply stash^{/${tmp_stash}}
git stash drop stash@{0}