#!/bin/bash
# ===========================================================================================================
# This script contains some functions to set up LPC environment for analysis quickly 
# ===========================================================================================================

function setup_dirname_local {
    export BASE_DIR=/uscms/home/joyzhou/work/hhbbtautau
    export LIB_DIR=${BASE_DIR}/scripts
    export SRC_DIR=${BASE_DIR}/src
    export DATA_DIR=${BASE_DIR}/data
    export CONDOR_BASE=/store/user/joyzhou
    export OUTPUT_BASE=/uscms/home/joyzhou/nobackup
    echo "Successfully set up environment or $PWD"
}

function human_readable() {
    size=$1
    units=("B" "KB" "MB" "GB" "TB" "PB" "EB" "ZB" "YB")
    for ((i=0; size>=1024 && i<${#units[@]}-1; i++)); do
        size=$(bc <<< "scale=2; $size/1024")
    done
    echo "$size${units[$i]}"
}

function setup_LCG {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        release_version=$VERSION_ID
    else
        echo "Cannot determine OS version"
        echo "Sourcing default centos 7 LCG setup"
        source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos7-gcc11-opt/setup.sh
        return 1
    fi

    if [[ "$release_version" == 8.* ]]; then
        echo "Performing operations for release 8.*"
        source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos8-gcc11-opt/setup.sh
    elif [[ "$release_version" == 9.* ]]; then
        echo "Performing operations for release 9.*"
        # do not change this, this will have impact on venv installment
        source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/setup.sh 
    else
        source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos7-gcc11-opt/setup.sh
    fi
}

function LPC_setup {
    export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch
    export EOS_MGM_URL=root://cmseos.fnal.gov
    source $VO_CMS_SW_DIR/cmsset_default.sh

    print_env_variable() { var="$1"; [ -z "${!var}" ] && echo "$var is not set" || echo "$var has been set to ${!var}"; }

    export PREFIX=root://cmseos.fnal.gov
    print_env_variable "PREFIX"

    alias condor_rm_held="condor_rm -constraint 'JobStatus == 5'"
    alias condor_rm_running="condor_rm -constraint 'JobStatus == 2'"
}

function LCG_sasetup {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        release_version=$VERSION_ID
    else
        echo "Cannot determine OS version"
        echo "Sourcing default centos 7 LCG setup"
        version=x86_64-centos7-gcc11-opt
    fi

    if [[ "$release_version" == 8.* ]]; then
        version=x86_64-centos7-gcc11-opt
    elif [[ "$release_version" == 9.* ]]; then
        version=x86_64-el9-gcc13-opt
    else
        version=x86_64-centos7-gcc11-opt
    fi

    echo $version

    source /cvmfs/sft.cern.ch/lcg/releases/LCG_105/Python/3.9.12/$version/Python-env.sh
    echo "Successfully sourced python package"
    source /cvmfs/sft.cern.ch/lcg/releases/LCG_105/ROOT/6.30.02/$version/ROOT-env.sh
    echo "Successfully sourced ROOT software"
    source /cvmfs/sft.cern.ch/lcg/releases/LCG_105/pyyaml/6.0.1/$version/PyYAML-env.sh
    echo "Successfully sourced PyYAML software"

}

function remove_duplicates {
    IFS=':' read -r -a array <<< "$1"
    declare -A seen
    result=()
    for i in "${array[@]}"; do
        if [[ ! -v seen[$i] ]]; then
            result+=("$i")
            seen["$i"]=1
        fi
    done
    new_path=$(IFS=":"; echo "${result[*]}")
    echo $new_path
}

function checkproxy {
    if [ -z "$X509_USER_PROXY" ]; then
        echo "Proxy not found. Please run `vominit` to create a proxy."
        exit 1
    else
        echo "Proxy found at $X509_USER_PROXY"
        export X509_CERT_DIR=/cvmfs/grid.cern.ch/etc/grid-security/certificates/
        voms-proxy-info -path -debug
        voms-proxy-info -file $X509_USER_PROXY -debug
        return 0
    fi
}

function vominit {
    voms-proxy-init --rfc --voms cms -valid 192:00
}

function checkvom {
    if voms-proxy-info --exists; then
    echo "VOMS proxy exists."

    if voms-proxy-info --valid 10 2>&1 > /dev/null; then
        echo "VOMS proxy is valid."
        timeleft=$(voms-proxy-info --timeleft)
        echo "Time left for the proxy: ${timeleft} seconds."
    else
        echo "VOMS proxy is not valid or has expired."
    fi
    else
    echo "No VOMS proxy found."
    fi
}

function cplocal {
    DIRNAME=$1
    echo "Copying from condor to local ........................"
    xrdcp -r root://cmseos.fnal.gov/${CONDOR_BASE}/${DIRNAME} ${OUTPUT_BASE}
}