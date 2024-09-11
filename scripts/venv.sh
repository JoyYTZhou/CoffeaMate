#!/bin/bash
# ===========================================================================================================
# This script sets up the environment to run this repository on lpc
# ===========================================================================================================

source scripts/envutil.sh
setup_LCG
LPC_setup
setup_dirname_local
ENV_NAME=skim_el9

export PYTHONPATH=~/nobackup/${ENV_NAME}/lib/python3.9/site-packages:$PYTHONPATH
source ~/nobackup/${ENV_NAME}/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH

export PATH=$(remove_duplicates "$PATH")
export PYTHONPATH=$(remove_duplicates "$PYTHONPATH")

echo "===================================="

if [ -z "$1" ]; then
  echo "Enter Dynaconf environment name: "
  read env_name
else
  env_name=$1
fi

export ENV_FOR_DYNACONF=$env_name
export DEBUG_MODE=false

alias jup='jupyter notebook --no-browser --port=2001'
