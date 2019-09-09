#!/usr/bin/env bash

python_version=$1
python_subversion=$2

which bigartm
if [[ $? -eq 0 ]]; then
    bigartm_bin=$(which bigartm)
    echo "Bigartm executable found at $bigartm_bin Skipping building and installing!"
    exit 0
fi

if [[ "$python_subversion" == "" ]]; then
    python_full_version="$python_version"
else
    python_full_version="$python_version.$python_subversion"
fi


which "python$python_full_version"
if [[ $? -eq 0 ]]; then
    python_bin=$(which "python$python_full_version")
    echo "Python executable found at $python_bin"
else
    echo "Python \"$python_full_version\" not found"
    which python3
    which python
    exit 1
fi

set -e

sudo apt-get --yes install python-setuptools python-wheel
sudo apt-get --yes install python3-setuptools python3-wheel


if [[ "$python_version" == "2" ]]; then
    python2 -m pip install --user -U setuptools wheel
    python3 -m pip install --user -U setuptools wheel
else
    python2 -m pip install --user -U setuptools wheel
    ${python_bin} -m pip install --user -U setuptools wheel
fi

sudo apt-get --yes install git make cmake build-essential libboost-all-dev gfortran libblas-dev liblapack-dev

current_dir=$(echo $PWD)
git clone https://github.com/bigartm/bigartm.git
mkdir build && cd build

# installs by default under /usr/local. To manipulate this use -DCMAKE_INSTALL_PREFIX=xxx flag in cmake
if [[ "$python_version" == "2" ]]; then
    cmake ..
elif [[ "$python_version" == "3" ]]; then
    cmake -DPYTHON=python3 ..
fi

make
sudo make install

# now the 'bigartm' executable should be accessible
which bigartm
if [[ $? -ne 0 ]]; then
    echo "Failed to find bigartm executable."
    cd ${current_dir}
    exit 1
fi

cd ${current_dir}
