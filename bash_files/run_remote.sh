#!/bin/bash

# check argument
if [ "$#" -ne 1 ]; then
    echo "no argument: tag missing"
	exit 0
fi

# CUSTOMIZE #
ssh_keyword=euler
#############

tmp_stash=seb_temp_stash
tmp_branch=seb_temp_branch

# update experiments files on git (make it temporary by defining a dedicated branch create delete)

git stash push -m ${tmp_stash} 
git checkout -b $tmp_branch
git stash apply stash^{/${tmp_stash}}
git add *
git commit -m "temporary commit on temporary branch for experiments '${1}' on Euler cluster"
git push --set-upstream origin $tmp_branch

# makes ssh connection and execute main commands 
# TODO: add the name of tmp branch to access for experiments
cat ./bash_files/main_commands.sh | ssh $ssh_keyword /bin/bash -s $1 "remote" $tmp_branch

# delete git branch, retrieve stashed changes and delete stash
git push origin --delete seb_temp_branch
git checkout master
git branch -D $tmp_branch
git stash apply stash^{/${tmp_stash}}
git stash drop stash@{0}