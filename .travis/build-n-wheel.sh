#!/usr/bin/env bash

set -e

python_version=$1
python_subversion=$2

if [[ "$python_version" == "2" ]]; then
    python_subversion=7
elif [[ "$python_subversion" == "" ]]; then
    echo "Please supply a 2nd argument to indicate specific python3 subversion (ie 6, 7)"
    exit 1
fi

python -m pip install "setuptools>=40.0.0"
python -m pip install -U setuptools
python -m pip install protobuf tqdm wheel

# sudo -H python -m pip install wheel

sudo apt-get --yes install python-setuptools python-wheel
sudo apt-get --yes install python3-setuptools python3-wheel

python -m pip list

current_dir=$(echo $PWD)
mkdir $BIGARTM_PARENT_DIR/bigartm/build
cd $BIGARTM_PARENT_DIR/bigartm/build

#sudo apt-get update --yes
sudo apt-get --yes install git make cmake build-essential libboost-all-dev gfortran libblas-dev liblapack-dev
#sudo apt-get upgrade --yes

# installs by default under /usr/local. To manipulate this use -DCMAKE_INSTALL_PREFIX=xxx flag in cmake
if [[ "$python_version" == "2" ]]; then
    cmake ..
elif [[ "$python_version" == "3" ]]; then
    cmake -DPYTHON=python3 ..
fi
echo DONE cmake


make
echo DONE make

sudo make install
echo DONE make install
export ARTM_SHARED_LIBRARY=/usr/local/lib/libartm.so

echo SETUPTOOLS
python -m pip list

# now the 'bigartm' executable should be accessible
which bigartm
if [[ $? -ne 0 ]]; then
    echo "Failed to find bigartm executable. Maybe not in path?"
    exit 1
fi


cd $current_dir
