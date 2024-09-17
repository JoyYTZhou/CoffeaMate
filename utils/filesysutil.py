import os, glob, tracemalloc, linecache, subprocess, datetime, fnmatch
from XRootD import client
from pathlib import Path

runcom = subprocess.run
pjoin = os.path.join
DEBUG_ON = os.environ.get("DEBUG_MODE", default=False)

def pbase(pathstr) -> str:
    """Get the base name of a path."""
    return pathstr.split('/')[-1]

def match(pathstr, pattern) -> bool:
    """Match a path with a pattern. Returns True if the path matches the pattern."""
    return fnmatch.fnmatch(pbase(pathstr), pattern)

class XRootDHelper:
    def __init__(self, prefix="root://cmseos.fnal.gov") -> None:
        self.xrdfs_client = client.FileSystem(prefix)

    def glob_files(self, dirname, filepattern="*", **kwargs) -> list:
        """Returns a list of files matching a pattern in a directory. By default will return all files."""
        status, listing = self.xrdfs_client.dirlist(dirname)
        if not status.ok:
            raise Exception(f"Failed to list directory {dirname}: {status.message}")
        if filepattern == '*':
            files = [entry.name for entry in listing.dirlist]
        else:
            files = [entry.name for entry in listing.dirlist if match(entry.name, filepattern)]
        return files
    
    def __check_path(self, dirname, createdir=True, raiseError=False) -> bool:
        """Check if a directory exists. If not will create one.
        
        Return 
        - True if the path exists, False otherwise"""
        status, _ = self.xrdfs_client.stat(dirname)
        if not status.ok:
            if raiseError:
                raise FileNotFoundError(f"this path {dirname} does not exist.")
            else:
                if createdir:
                    status, _ = self.xrdfs_client.mkdir(dirname)
                    if not status.ok:
                        raise Exception(f"Failed to create directory {dirname}: {status.message}")
                return False
        return True
    
    def remove_files(self, dirname, pattern='*') -> None:
        """Delete all files in a directory with a specific pattern."""
        files = self.glob_files(dirname, pattern)
        for file in files:
            status, _ = self.xrdfs_client.rm(pjoin(dirname, file))
            if not status.ok:
                raise Exception(f"Failed to remove {file}: {status.message}")
    
    def transfer_files(self, srcpath, destpath, filepattern='*', remove=False) -> None:
        """Transfer all files matching filepattern from srcpath to destpath. Will create the destpath if it doesn't exist.
        This is only meant for transferring files from local to xrdfs."""

        if srcpath.startswith('/store/user'):
            raise ValueError("Source path should be a local directory. Why are you transferring from one EOS to another?")
        else:
            files = glob.glob(pjoin(srcpath, filepattern))

        status = self.__check_path(destpath)
        for file in files:
            src_file = file
            dest_file = pjoin(destpath, file)
            status, _ = self.xrdfs_client.copy(src_file, dest_file)
            if not status.ok:
                raise Exception(f"Failed to copy {src_file} to {dest_file}: {status.message}")
            if remove:
                os.remove(src_file)
                if not status.ok:
                    raise Exception(f"Failed to remove {src_file}: {status.message}")

def glob_files(dirname, filepattern='*', **kwargs) -> list:
    """Returns a SORTED list of files matching a pattern in a directory. By default will return all files.
    
    Parameters
    - `dirname`: directory path (remote/local)
    - `filepattern`: pattern to match the file name. Wildcards allowed
    - `kwargs`: additional arguments for filtering files

    Return
    - A SORTED list of files (str)
    """
    if dirname.startswith('/store/user'):
        files = get_xrdfs_files(dirname, filepattern, **kwargs)
    else:
        if filepattern == '*':
            files = [str(file.absolute()) for file in Path(dirname).iterdir() if file.is_file()]
        else:
            files = glob.glob(pjoin(dirname, filepattern)) 
    return sorted(files)

def checkpath(dirname, createdir=True, raiseError=False) -> bool:
    """Check if a directory exists. If not will create one.
    
    Return:
    - True if the path exists, False otherwise"""
    if dirname.startswith('/store/user/'):
        return checkcondorpath(dirname, createdir, raiseError=raiseError)
    else:
        return checklocalpath(dirname, createdir, raiseError)

def transferfiles(srcpath, destpath, filepattern=False, remove=False) -> None:
    """Transfer all files in srcpath to destpath. Will check if destpath exist.
    
    Parameters
    - `srcpath`: source path (local/remote), a directory
    - `destpath`: destination path (remote/local), a directory
    - `filepattern`: 
    - `remove`: mv/cp. Only supported when moving from local to remote
    """
    if isremote(destpath):
        if isremote(srcpath):
            raise ValueError("Source path should be a local directory. Why are you transferring from one EOS to another?")
        else:
            checkcondorpath(destpath)
            if filepattern:
                for srcfile in glob_files(srcpath, filepattern):
                    cpcondor(srcfile, f'{destpath}/{os.path.basename(srcfile)}')
                    if remove: os.remove(srcfile)
            else:
                cpcondor(srcpath, destpath)
                if remove: os.remove(srcpath)
    elif isremote(srcpath):
        if isremote(destpath):
            raise ValueError("Destination path should be a local directory. Why are you transferring from EOS to EOS?")
        else:
            checklocalpath(destpath)
            if filepattern:
                for srcfile in glob_files(srcpath, filepattern):
                    cpfcondor(srcfile, f'{destpath}/')
            else: cpfcondor(srcpath, f'{destpath}/')

def remove_xrdfs_file(file_path):
    """Remove a file from xrdfs.
    
    Parameters
    - `file_path`: file path to remove. without prefix"""
    cmd = ["xrdfs", PREFIX, "rm", file_path]
    subprocess.run(cmd)

def checkx509():
    """Check if the X509 proxy and certificate directory are set."""
    proxy_position = os.environ.get("X509_USER_PROXY", default="None")
    if proxy_position is None:
        raise SystemError("Proxy not found. Immediately check proxy!")
    print(f"Proxy at: {proxy_position}.")
    proxy_directory = os.environ.get("X509_CERT_DIR", default="None")
    if proxy_directory is None:
        raise SystemError("Certificate directory not set. Immmediately check certificate directory!")
    print(f"Certificate directory set to be {proxy_directory}.")

def display_top(snapshot, key_type='lineno', limit=10):
    """Display the top lines of a snapshot"""
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))
    
def checklocalpath(pathstr, createdir=True, raiseError=False) -> bool:
    """Check if a local path exists. If not will create one.
    
    Return
    - True if the path exists, False otherwise"""
    path = Path(pathstr) 
    if not path.exists():
        if raiseError:
            raise FileNotFoundError(f"this path {pathstr} does not exist.")
        else:
            if createdir: path.mkdir(parents=True, exist_ok=True)
        return False
    return True

def cpfcondor(srcpath, localpath):
    """Copy a root file FROM condor to LOCAL."""
    comstr = f'xrdcp {srcpath} {localpath}' if srcpath.startswith('root://') else f'xrdcp {PREFIX}/{srcpath} {localpath}'
    result = runcom(comstr, shell=True, capture_output=True, text=True)
    if DEBUG_ON and result.returncode !=0 : print(result.stderr)
    return result

def cpcondor(srcpath, destpath):
    """Copy srcpath (file/directory) FROM local to condor destpath"""
    comstr = f'xrdcp {srcpath} {destpath}' if destpath.startswith('root://') else f'xrdcp {srcpath} {PREFIX}/{destpath}' 
    result = runcom(comstr, shell=True, capture_output=True, text=True)
    if DEBUG_ON and result.returncode !=0 : print(result.stderr)
    return result

def isremote(pathstr):
    """Check if a path is remote."""
    is_remote = pathstr.startswith('/store/user') or pathstr.startswith("root://")
    return is_remote

def cross_check(filepattern, existentfiles) -> bool:
    """Check if a file pattern exists in a list of files.
    
    Parameters
    - `filepattern`: pattern to match the file name
    - `existentfiles`: list of files to check
    
    Return
    - bool: True if the file pattern exists in the list of files
    """
    for file in existentfiles:
        basename = file.split('/')[-1]
        if fnmatch.fnmatch(basename, filepattern):
            return True
    return False

def checkcondorpath(dirname, createdir=True, raiseError=False) -> bool:
    """Check if a condor path exists and potentially create one.

    Parameters:
    `dirname`: directory path to check
    `createdir`: if True, create the directory if it doesn't exist. If false, return stat code. 
    `raiseError`: if True, raise an error if the directory doesn't exist.
    
    Return: 
    - bool: True if the directory exists, False otherwise."""
    check_dir_cmd = f"xrdfs {PREFIX} stat {dirname}"
    create_dir_cmd = f"xrdfs {PREFIX} mkdir -p {dirname}"
    proc = runcom(check_dir_cmd, shell=True, capture_output=True, text=True, check=raiseError) 
    if proc.returncode != 0:
        if createdir:
            runcom(create_dir_cmd, shell=True)
        else:
            return False
    else:
        return True

def delfiles(dirname, pattern='*.root'):
    """Delete all files in a directory with a specific pattern."""
    if pattern is not None:
        dirpath = Path(dirname)
        for fipath in dirpath.glob(pattern):
            fipath.unlink()

def get_xrdfs_files(remote_dir, filepattern='*', add_prefix=True) -> list[str]:
    """Get the files in a remote directory that match a pattern. If both patterns==None, returns all files.
    
    Parameters:
    - `remote_dir`: remote directory path
    - `filepattern`: pattern to match the file name. Wildcards (*, ?) allowed
    - `add_prefix`: if True, will add the PREFIX to the file path
    
    Returns:
    - list of files that match the pattern
    """
    all_files = list_xrdfs_files(remote_dir)
    if filepattern == '*':
        return sorted(all_files)
    else:
        if add_prefix:
            filtered_files = [PREFIX + "/" + f for f in all_files if fnmatch.fnmatch(f.split('/')[-1], filepattern)]
        else: 
            filtered_files = [f for f in all_files if fnmatch.fnmatch(f.split('/')[-1], filepattern)]
        return sorted(filtered_files)

def list_xrdfs_files(remote_dir) -> list[str]:
    """List files/dirs in a remote xrdfs directory using subprocess.run."""
    cmd = ["xrdfs", PREFIX, "ls", remote_dir]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files = sorted(result.stdout.strip().split('\n'))
    return files

def get_xrdfs_file_info(remote_file, redir=PREFIX):
    """Get information (size, modification time) of a remote xrdfs file/dir.
    
    Parameters
    ``remote_file``: remote file path
    Returns
    - size of the file in bytes (int)
    - modification time of the file (str)"""
    cmd = ["xrdfs", redir, "stat", remote_file]
    output = subprocess.check_output(cmd).decode()
    size = None
    mod_time = None

    for line in output.split('\n'):
        if line.startswith('Size:'):
            size = int(line.split()[1])
        elif line.startswith('MTime:'):
            mod_time = ' '.join(line.split()[1:])
    return size, mod_time

def sync_files(local_dir, remote_dir):
    """Check for discrepancies and update local files from a remote xrdfs directory."""
    remote_files = list_xrdfs_files(remote_dir)
    for remote_file in remote_files:
        local_file = os.path.join(local_dir, os.path.basename(remote_file))
        if not os.path.exists(local_file):
            copy_file = True
        else:
            local_size = os.path.getsize(local_file)
            local_mod_time = datetime.fromtimestamp(os.path.getmtime(local_file))
            remote_size, remote_mod_time_str = get_xrdfs_file_info(remote_file)
            remote_mod_time = datetime.strptime(remote_mod_time_str, '%Y-%m-%d %H:%M:%S')
            copy_file = (local_size != remote_size) or (local_mod_time < remote_mod_time)
        
        if copy_file:
            cmd = ["xrdcp", f"{PREFIX}/{remote_file}", local_file]
            subprocess.run(cmd)