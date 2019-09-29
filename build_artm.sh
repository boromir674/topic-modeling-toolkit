#!/usr/bin/env bash


usage() {
    echo "Usage:"
    echo " $0 [ --bigartm <value> ] [ --DCMAKE_INSTALL_PREFIX <value> ] [ --python <value> ]"
    echo " $0 [ --help | -h ]"
    echo
    echo "Default bigartm-folder is 'bigartm'. Default DCMAKE_INSTALL_PREFIX is '/usr/local'. Default python-version is '3'."
}

# set defaults
bigartm_folder=bigartm
DCMAKE_INSTALL_PREFIX=/usr/local  # /opt/bigartm
python_version=3
# DBUILD_BIGARTM_CLI_STATIC=ON – to use static versions of Boost, C and C++ libraries for


i=$(($# + 1)) # index of the first non-existing argument
declare -A longoptspec
# Use associative array to declare how many arguments a long option
# expects. In this case we declare that loglevel expects/has one
# argument and range has two. Long options that aren't listed in this
# way will have zero arguments by default.
longoptspec=( [bigartm]=1 [DCMAKE_INSTALL_PREFIX]=1 [python]=1 )
optspec=":l:h-:"
while getopts "$optspec" opt; do
while true; do
    case "${opt}" in
        -) #OPTARG is name-of-long-option or name-of-long-option=value
            if [[ ${OPTARG} =~ .*=.* ]] # with this --key=value format only one argument is possible
            then
                opt=${OPTARG/=*/}
                ((${#opt} <= 1)) && {
                    echo "Syntax error: Invalid long option '$opt'" >&2
                    exit 2
                }
                if (($((longoptspec[$opt])) != 1))
                then
                    echo "Syntax error: Option '$opt' does not support this syntax." >&2
                    exit 2
                fi
                OPTARG=${OPTARG#*=}
            else #with this --key value1 value2 format multiple arguments are possible
                opt="$OPTARG"
                ((${#opt} <= 1)) && {
                    echo "Syntax error: Invalid long option '$opt'" >&2
                    exit 2
                }
                OPTARG=(${@:OPTIND:$((longoptspec[$opt]))})
                ((OPTIND+=longoptspec[$opt]))
                #echo $OPTIND
                ((OPTIND > i)) && {
                    echo "Syntax error: Not all required arguments for option '$opt' are given." >&2
                    exit 3
                }
            fi

            continue #now that opt/OPTARG are set we can process them as
            # if getopts would've given us long options
            ;;
        bigartm)
            bigartm_folder=$OPTARG
            ;;
	    DCMAKE_INSTALL_PREFIX)
            DCMAKE_INSTALL_PREFIX=$OPTARG
            ;;
        python)
            python_version=$OPTARG
            ;;
        h|help)
            usage
            exit 0
            ;;
        ?)
            echo "Syntax error: Unknown short option '$OPTARG'" >&2
            exit 2
            ;;
        *)
            echo "Syntax error: Unknown long option '$opt'" >&2
            exit 2
            ;;
    esac
break; done
done

echo "bigartm_folder: $bigartm_folder"
echo "DCMAKE_INSTALL_PREFIX: $DCMAKE_INSTALL_PREFIX"
echo "python_version: $python_version"

#echo "First non-option-argument (if exists): ${!OPTIND-}"
##echo "Second non-option-argument (if exists): ${!OPTIND-}"
#
#shift "$((OPTIND-1))"   # Discard the options and sentinel --
#printf '<%s>\n' "$@"


which bigartm
if [[ $? -eq 0 ]]; then
    bigartm_bin=$(which bigartm)
    echo "Bigartm executable found at $bigartm_bin Skipping building and installing!"
    exit 0
fi


which "python$python_version"
if [[ $? -eq 0 ]]; then
    python_bin=$(which "python$python_version")
    echo "Python executable found at $python_bin"
else
    echo "Executable 'python$python_version' not found"
    which python3
    which python
    exit 1
fi

set -e

sudo apt-get --yes install python-setuptools python-wheel
sudo apt-get --yes install python3-setuptools python3-wheel


${python_bin} -m pip install --user -U setuptools wheel


sudo apt-get --yes install git make cmake build-essential libboost-all-dev gfortran libblas-dev liblapack-dev

current_dir=$(echo $PWD)
git clone https://github.com/bigartm/bigartm.git $bigartm_folder
cd $bigartm_folder

mkdir build && cd build

# installs by default under /usr/local. To manipulate this use -DCMAKE_INSTALL_PREFIX=xxx flag in cmake
if [[ "$python_version" == "2" ]]; then
    cmake -DCMAKE_INSTALL_PREFIX=$DCMAKE_INSTALL_PREFIX ..
elif [[ "$python_version" == "3" ]]; then
    cmake -DCMAKE_INSTALL_PREFIX=$DCMAKE_INSTALL_PREFIX -DPYTHON=python3 ..
fi

make
sudo make install

echo "Created wheel(s)"
ls python/bigartm*.whl

export ARTM_SHARED_LIBRARY="$DCMAKE_INSTALL_PREFIX/lib/libartm.so" && echo "Exported ARTM_SHARED_LIBRARY=$DCMAKE_INSTALL_PREFIX/lib/libartm.so"

# now the 'bigartm' executable should be accessible
which bigartm
if [[ $? -ne 0 ]]; then
    echo "Failed to find bigartm executable."
    cd ${current_dir}
    exit 1
fi
echo "Executable 'bigartm' installed"
cd ${current_dir}
