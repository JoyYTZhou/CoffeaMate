#!/bin/bash

pwd
echo "List all files in $OUTPUTPATH"
ls -lR $OUTPUTPATH
echo "*******************************************"
echo "xrdcp output for condor to "
echo $CONDORPATH

src=$OUTPUTPATH
dest=$CONDORPATH/$PROCESS_NAME

find "$src" -type f | while read -r file; do
    relpath="${file#$src}"
    xrdcp -f "$file" "$dest$relpath"
done

rm -r $OUTPUTPATH
