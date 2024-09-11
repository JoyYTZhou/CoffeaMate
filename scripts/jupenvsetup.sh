#!/usr/bin/bash
# ===========================================================================================================
# This script sets up a proper virtual environment for jupyter notebook in the ~/nobackup area
# ===========================================================================================================
source scripts/envutil.sh
setup_LCG

cd ~/nobackup
export ENV_NAME=coffeajup_el9
python -m venv ${ENV_NAME}
echo "creating new venv..."
source ${ENV_NAME}/bin/activate

python3 -m pip install coffea --upgrade --no-cache-dir
echo "Installed Coffea"
python3 -m pip install dynaconf --no-cache-dir
echo "Installed dyanconf"
python3 -m pip install vector --upgrade --no-cache-dir
echo "Installed vector"
python3 -m pip install dask --upgrade --no-cache-dir
python3 -m pip install awkward --upgrade --no-cache-dir
python3 -m pip install dask_awkward --upgrade --no-cache-dir
python3 -m pip install hist --upgrade --no-cache-dir
deactivate


