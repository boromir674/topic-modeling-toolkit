#!/usr/bin/env bash

python_version=$1
python_subversion=$2

if [[ "$python_version" == "2" ]]; then
    python_subversion=7
elif [[ "$python_subversion" == "" ]]; then
    echo "Please supply a 2nd argument to indicate specific python3 subversion (ie 6, 7)"
    exit 1
fi

reg="bigartm.*cp$python_version$python_subversion.*.whl"

for filename in $BIGARTM_PARENT_DIR/bigartm/build/python/bigartm*.whl; do
    if [[ $filename =~ $reg ]]; then
        echo $filename
#  	    echo "Found wheel for python$python_version.$python_subversion"
#	    export BIGARTM_WHEEL=$filename
	    break
	fi
done
