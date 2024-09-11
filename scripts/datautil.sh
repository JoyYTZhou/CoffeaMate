#!/bin/bash

function check_file {
    FILENAME=$1
    dasgoclient --query="site file=${FILENAME}"
    xrdfs root://cmsxrootd.fnal.gov/ ls -l $FILENAME
    xrdfs cms-xrd-global.cern.ch locate $FILENAME
}

function check_size {
    REDIRECTOR=$1
    FILENAME=$2
    output=$(xrdfs $REDIRECTOR stat $FILENAME)
    size=$(echo "$output" | grep -oP 'Size:\s+\K\d+')
}

function eosbackup {
    echo "Enter the directory name to backup"
    read DIRNAME
    echo "===================================="
    echo "The following directory will be copied"
    echo /store/user/joyzhou/$DIRNAME
    export EOS_MGM_URL=root://cmseos.fnal.gov
    eos cp -r /eos/uscms/store/user/joyzhou/$DIRNAME/ /eos/uscms/store/user/joyzhou/backup/$DIRNAME/ >> eosbackup.log & 
}
