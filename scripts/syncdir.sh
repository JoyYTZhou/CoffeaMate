#!/bin/bash
# testing needed

# Define remote and local directories
REMOTE_DIR="root://server.example.com/path/to/remote/dir"
LOCAL_DIR="/path/to/local/dir"

# List files in remote directory
remote_files=$(xrdfs server.example.com ls $REMOTE_DIR)

# Iterate over remote files to check and sync them
for remote_file in $remote_files; do
    # Extract filename from remote_file path
    filename=$(basename $remote_file)

    # Define local file path
    local_file="${LOCAL_DIR}/${filename}"

    # Check if file exists locally and compare sizes if it does
    if [[ -e "$local_file" ]]; then
        # Get local and remote file sizes
        local_size=$(stat -c%s "$local_file")
        remote_size=$(xrdfs server.example.com stat "$remote_file" | grep Size | awk '{print $2}')

        # If sizes do not match, or remote file is newer, copy the file
        if [[ "$local_size" != "$remote_size" ]]; then
            echo "Updating $filename..."
            xrdcp "$REMOTE_DIR/$filename" "$local_file"
        fi
    else
        # If file does not exist locally, copy it
        echo "Copying new file $filename..."
        xrdcp "$REMOTE_DIR/$filename" "$local_file"
    fi
done

echo "Synchronization complete."
