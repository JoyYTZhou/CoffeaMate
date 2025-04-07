import os, glob, shutil, tracemalloc, linecache, subprocess, fnmatch, psutil, logging, time, gc, mmap, sys
from datetime import datetime
from XRootD import client
from pathlib import Path
from src.utils.displayutil import display_directory_stats

runcom = subprocess.run
pjoin = os.path.join
pdir = os.path.dirname
DEBUG_ON = os.environ.get("DEBUG_MODE", default=False)
PREFIX = "root://cmseos.fnal.gov"

def pbase(pathstr) -> str:
    """Get the base name of a path."""
    return pathstr.split('/')[-1]

def match(pathstr, pattern) -> bool:
    """Match a path with a pattern. Returns True if the path matches the pattern."""
    return fnmatch.fnmatch(pbase(pathstr), pattern)

def release_mapped_memory():
    """Find and unmap memory-mapped files in Python."""
    gc.collect()
    for obj in gc.get_objects():
        if isinstance(obj, mmap.mmap):
            obj.close()
            logging.debug(f"Unmapped memory-mapped file {obj}")
            time.sleep(1)

def is_remote(filepath: str) -> bool:
    """Check if a file path is remote (XRootD) or local.
    
    Parameters
    - filepath: path to check
    
    Returns
    - bool: True if the path is remote (starts with 'root://' or '/store/user'), False otherwise
    """
    return filepath.startswith('root://') or filepath.startswith('/store/user')

def strip_xrd_prefix(filepath: str) -> str:
    """Strip the XRootD prefix (root://*/) from a file path.
    
    Parameters
    - filepath: full file path including XRootD prefix
    
    Returns
    - str: file path without the XRootD prefix
    
    Example:
    >>> strip_xrd_prefix("root://cmseos.fnal.gov//store/user/file.root")
    '/store/user/file.root'
    """
    if filepath.startswith('root://'):
        # Find the index after the host name (after the next '/' after '//')
        double_slash_idx = filepath.find('//')
        if double_slash_idx != -1:
            next_slash_idx = filepath.find('/', double_slash_idx + 2)
            if next_slash_idx != -1:
                return filepath[next_slash_idx:]
    return filepath

class FileSysHelper:
    """Helper class for file system operations. Can be used for both local and remote file systems."""
    def __init__(self) -> None:
        pass

    @staticmethod
    def close_open_files_delete(dirname, pattern, max_retries=3, wait_time=1):
        """Close all open files in a directory with a specific pattern and then delete them.

        Parameters
        - `dirname`: directory path
        - `pattern`: pattern to match the file name
        - `max_retries`: maximum number of attempts to close a file
        - `wait_time`: time to wait between attempts (in seconds)
        """
        process = psutil.Process()
        open_files = {file.path for file in process.open_files()}
        to_delete = glob.glob(pjoin(dirname, pattern))

        for file in to_delete:
            logging.info(f"Checking if file {file} is open...")
            if file in open_files:
                logging.warning(f"File {file} is open. Attempting to close...")

                for attempt in range(max_retries):
                    try:
                        open(file, 'r').close()
                        # Check if file is still open
                        current_open = {f.path for f in process.open_files()}
                        if file not in current_open:
                            logging.info(f"Successfully closed file {file}")
                            break
                        logging.warning(f"Attempt {attempt + 1}/{max_retries}: File {file} still open, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    except Exception as e:
                        logging.exception(f"Attempt {attempt + 1}/{max_retries}: Failed to close file {file}: {str(e)}")
                        if attempt < max_retries - 1:  # Don't sleep on last attempt
                            time.sleep(wait_time)
                if file in {f.path for f in process.open_files()}:
                    logging.error(f"Failed to close file {file} after {max_retries} attempts")
            os.remove(file)
            logging.debug(f"Deleted file {file}")

    @staticmethod
    def remove_emptydir(root_dir):
        for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
            if not dirnames and not filenames:
                os.rmdir(dirpath)

    @staticmethod
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

    @staticmethod
    def glob_files(dirname, filepattern='*', full_path=True, exclude=None, **kwargs) -> list:
        """Returns a SORTED list of files matching a pattern in a directory. By default will return all files.
        
        Parameters
        - `dirname`: directory path (remote/local)
        - `filepattern`: pattern to match the file name. Wildcards allowed
        - `full_path`: whether to return the full path or with just file name globbed
        - `exclude`: pattern or list of patterns to exclude. Wildcards allowed
        - `kwargs`: additional arguments for filtering files

        Return
        - A SORTED list of files (str)
        """
        if is_remote(dirname):
            xrdhelper = XRootDHelper(kwargs.get("prefix", PREFIX))
            files = xrdhelper.glob_files(dirname, filepattern, full_path, exclude=exclude)
        else:
            if filepattern == '*':
                files = [str(file.absolute()) for file in Path(dirname).iterdir() if file.is_file()]
            else:
                files = glob.glob(pjoin(dirname, filepattern))
            
            # Handle exclusions for local files
            if exclude:
                if isinstance(exclude, str):
                    exclude = [exclude]
                filtered_files = []
                for file in files:
                    basename = pbase(file)
                    if not any(fnmatch.fnmatch(basename, exc_pattern) for exc_pattern in exclude):
                        filtered_files.append(file)
                files = filtered_files

        return sorted(files)
    
    @staticmethod
    def glob_subdirs(dirname, dirpattern='*', full_path=True, **kwargs) -> list:
        """Returns a SORTED list of subdirectories matching a pattern in a directory. By default will return all subdirectories.
        
        Return 
        - A SORTED list of subdirectories (str)"""
        if dirname.startswith('/store/user'):
            xrdhelper = XRootDHelper(kwargs.get("prefix", PREFIX))
            subdirs = xrdhelper.glob_files(dirname, dirpattern, full_path)
        else:
            if dirpattern == '*':
                subdirs = [str(file.absolute()) for file in Path(dirname).iterdir() if file.is_dir()]
            else:
                subdirs = glob.glob(pjoin(dirname, dirpattern))
        return sorted(subdirs)
    
    @staticmethod
    def checkpath(pathstr, createdir=True, raiseError=False, prefix=PREFIX) -> bool:
        """Check if a path exists. If not will create one.
        
        Return
        - True if the path exists, False otherwise"""
        if pathstr.startswith('/store/user'):
            xrdhelper = XRootDHelper(prefix)
            return xrdhelper.check_path(pathstr, createdir, raiseError)
        else:
            path = Path(pathstr)
            if not path.exists():
                if raiseError:
                    raise FileNotFoundError(f"this path {pathstr} does not exist.")
                else:
                    if createdir: path.mkdir(parents=True, exist_ok=True)
                return False
            return True
    
    @staticmethod
    def remove_files(dirname, pattern='*', prefix=PREFIX) -> None:
        """Delete all files in a directory with a specific pattern.
        
        Parameters
        - `dirname`: directory path (remote/local)
        - `pattern`: pattern to match the file name. Wildcards allowed
        """
        if dirname.startswith('/store/user'):
            xrdhelper = XRootDHelper(prefix)
            xrdhelper.remove_files(dirname, pattern)
        else:
            files = glob.glob(pjoin(dirname, pattern))
            for file in files:
                os.remove(file)
        
    def remove_filelist(filelist, prefix=PREFIX) -> None:
        """Delete a list of files.
        
        Parameters
        - `filelist`: list of files to delete
        """
        if is_remote(filelist[0]):
            xrdhelper = XRootDHelper(prefix)
            for file in filelist:
                file = strip_xrd_prefix(file)
                if not xrdhelper.check_path(file, createdir=False, raiseError=False):
                    logging.warning(f"File {file} does not exist. Skipping.")
                    continue
                status, _ = xrdhelper.xrdfs_client.rm(file)
                if not status.ok:
                    raise Exception(f"Failed to remove {file}: {status.message}")
        else:
            for file in filelist:
                os.remove(file)

    @staticmethod
    def transfer_files(srcpath, destpath, filepattern='*', remove=False, overwrite=False, **kwargs) -> None:
        """Transfer all files matching filepattern from srcpath to destpath. Will create the destpath if it doesn't exist.
        
        Parameters 
        - `srcpath`: source path (local), a directory
        - `destpath`: destination path (remote), a directory
        - `filepattern`: pattern to match the file name. Passed into glob.glob(filepattern)
        - `remove`: whether to remove the files from srcpath after transferring
        - `overwrite`: whether to overwrite the files in the destination
        """
        if is_remote(destpath):
            xrdhelper = XRootDHelper(kwargs.get("prefix", PREFIX))
            xrdhelper.transfer_files(srcpath, destpath, filepattern, remove, overwrite)
        else:
            files = glob.glob(pjoin(srcpath, filepattern))
            if not os.path.exists(destpath):
                os.makedirs(destpath, exist_ok=True)
            for file in files:
                dest_file = pjoin(destpath, pbase(file))
                if not os.path.exists(dest_file) or overwrite:
                    shutil.copy(file, dest_file)
                    if remove:
                        os.remove(file)
                else:
                    logging.debug(f"File {dest_file} exists. Skipping.")

    @staticmethod
    def query_directory_structure(base_dir, prefix=PREFIX):
        """Query a directory and return a nested dictionary of its structure.

        Returns a dictionary with structure:
        {year: {groupname: {"lastUpdated": timestamp, "size": Megabytes}}}
        """
        if is_remote(base_dir):
            xrdhelper = XRootDHelper(prefix)
            clean_base = strip_xrd_prefix(base_dir)
            status, years = xrdhelper.xrdfs_client.dirlist(clean_base)
            if not status.ok:
                raise Exception(f"Failed to list directory {base_dir}: {status.message}")

            result = {}
            for year in years.dirlist:
                year_path = f"{clean_base}/{year.name}"
                status, groups = xrdhelper.xrdfs_client.dirlist(year_path)
                if not status.ok:
                    continue

                result[year.name] = {}
                for group in groups.dirlist:
                    group_path = f"{year_path}/{group.name}"
                    status, stat_info = xrdhelper.xrdfs_client.stat(group_path)
                    if not status.ok:
                        continue

                    size = stat_info.size/1024/1024 # Convert to MB

                    result[year.name][group.name] = {
                        "lastUpdated": stat_info.modtime,
                        "size": size
                    }
            return result
        else:
            result = {}
            for year in os.listdir(base_dir):
                year_path = pjoin(base_dir, year)
                if not os.path.isdir(year_path):
                    continue

                result[year] = {}
                for group in os.listdir(year_path):
                    group_path = pjoin(year_path, group)
                    if not os.path.isdir(group_path):
                        continue

                    result[year][group] = {
                        "lastUpdated": os.path.getmtime(group_path),
                        "size": sum(os.path.getsize(pjoin(dirpath, f))
                                for dirpath, _, filenames in os.walk(group_path)
                                for f in filenames)
                    }
            return result
 
    @staticmethod
    def get_file_size(filepath, prefix=PREFIX) -> int:
        """Get the size of a file in bytes. Works for both local and remote files.
        
        Parameters
        - filepath: path to the file (local or remote)
        - prefix: XRootD prefix for remote files (default: root://cmseos.fnal.gov)
        
        Returns
        - int: size of the file in bytes
        
        Raises
        - FileNotFoundError: if the file doesn't exist
        - Exception: if there's an error getting the size
        """
        if is_remote(filepath):
            xrdhelper = XRootDHelper(prefix)
            clean_path = strip_xrd_prefix(filepath)
            return xrdhelper.get_file_size(clean_path)
        else:
            return os.path.getsize(filepath) 
           
class XRootDHelper:
    def __init__(self, prefix=PREFIX) -> None:
        self.xrdfs_client = client.FileSystem(prefix)
        self.prefix = prefix

    def glob_files(self, dirname, filepattern="*", full_path=True, **kwargs) -> list:
        """Returns a list of files matching a pattern in a directory. By default will return all files/subdirectories."""
        exist = self.check_path(dirname, createdir=False, raiseError=False)
        if exist == False:
            return []
        status, listing = self.xrdfs_client.dirlist(dirname)
        if not status.ok:
            raise Exception(f"Failed to list directory {dirname}: {status.message}")
        if filepattern == '*':
            files = [entry.name for entry in listing.dirlist]
        else:
            files = [entry.name for entry in listing.dirlist if match(entry.name, filepattern)]
        if full_path:
            files = [f'{self.prefix}/{os.path.join(dirname, f)}' for f in files]
        return files
    
    def glob_files(self, dirname, filepattern="*", full_path=True, exclude=None, **kwargs) -> list:
        """Returns a list of files matching a pattern in a directory. By default will return all files/subdirectories.
        
        Parameters
        - dirname: directory path
        - filepattern: pattern to match the file name
        - full_path: whether to return full paths
        - exclude: pattern or list of patterns to exclude
        """
        clean_dirname = strip_xrd_prefix(dirname)
        exist = self.check_path(clean_dirname, createdir=False, raiseError=False)
        if exist == False:
            return []
            
        status, listing = self.xrdfs_client.dirlist(clean_dirname)
        if not status.ok:
            raise Exception(f"Failed to list directory {clean_dirname}: {status.message}")
        
        # First apply inclusion pattern
        if filepattern == '*':
            files = [entry.name for entry in listing.dirlist]
        else:
            files = [entry.name for entry in listing.dirlist if match(entry.name, filepattern)]
        
        # Then apply exclusion patterns
        if exclude:
            if isinstance(exclude, str):
                exclude = [exclude]
            filtered_files = []
            for file in files:
                if not any(fnmatch.fnmatch(file, exc_pattern) for exc_pattern in exclude):
                    filtered_files.append(file)
            files = filtered_files
        
        if full_path:
            files = [f'{self.prefix}/{os.path.join(clean_dirname, f)}' for f in files]
        
        return files

    def get_file_size(self, filepath) -> int:
        """Get the size of a file in bytes.
        
        Parameters
        - filepath: path to the file (remote), must start with /store/user/...
        
        Returns
        - int: size of the file in bytes
        
        Raises
        - Exception: if the file doesn't exist or there's an error getting the size
        """
        status, stat_info = self.xrdfs_client.stat(filepath)
        if not status.ok:
            raise Exception(f"Failed to get file size for {filepath}: {status.message}")
        return stat_info.size
    
    def check_path(self, dirname, createdir=True, raiseError=False) -> bool:
        """Check if a directory/file exists. If not will create directory of the same name (DO NOT USE FOR FILES).
        
        Return 
        - True if the path exists, False otherwise"""
        status, _ = self.xrdfs_client.stat(dirname)
        if not status.ok:
            if raiseError:
                raise FileNotFoundError(f"this path {dirname} does not exist.")
            else:
                logging.warning(f"Path {dirname} does not exist.")
                if createdir:
                    status, _ = self.xrdfs_client.mkdir(dirname)
                    if not status.ok:
                        raise Exception(f"Failed to create directory {dirname}: {status.message}")
                return False
        return True
    
    def remove_files(self, dirname, pattern='*') -> None:
        """Delete all files in an xrd directory with a specific pattern."""
        exist = self.check_path(dirname, createdir=False, raiseError=False)
        if exist == False:
            return
        files = self.glob_files(dirname, pattern, full_path=False)
        for file in files:
            status, _ = self.xrdfs_client.rm(pjoin(dirname, file))
            if not status.ok:
                raise Exception(f"Failed to remove {file}: {status.message}")
 

    @staticmethod 
    def call_xrdcp(src_file, dest_file, prefix=PREFIX):
        status = runcom(f'xrdcp {src_file} {prefix}/{dest_file}', shell=True, capture_output=True)
        if status.returncode != 0:
            raise Exception(f"Failed to copy {src_file} to {dest_file}: {status.stderr}")
    
    @staticmethod
    def copy_local(src_file, dest_file):
        status = runcom(f'xrdcp {src_file} {dest_file}', shell=True, capture_output=True)
        if status.returncode != 0:
            raise Exception(f"Failed to copy {src_file} to {dest_file}: {status.stderr}")
    
    def transfer_files(self, srcpath, destpath, filepattern='*', remove=False, overwrite=True) -> None:
        """Transfer all files matching filepattern from srcpath to destpath. Will create the destpath if it doesn't exist.
        This is only meant for transferring files from local to xrdfs.
        
        Parameters 
        - `srcpath`: source path (local), a directory
        - `destpath`: destination path (remote), a directory
        - `filepattern`: pattern to match the file name. Passed into glob.glob(filepattern)
        - `remove`: whether to remove the files from srcpath after transferring"""
        if is_remote(srcpath):
            raise ValueError("Source path should be a local directory. Why are you transferring from one EOS to another?")
        else:
            files = glob.glob(pjoin(srcpath, filepattern))
        
        if not is_remote(destpath):
            raise ValueError("Destination path should be a remote directory. Use FileSysHelper for local transfers.")
        
        destpath = strip_xrd_prefix(destpath)
        self.check_path(destpath)

        for file in files:
            src_file = file
            dest_file = pjoin(destpath, pbase(file))
            status, _ = self.xrdfs_client.stat(dest_file)
            if status.ok:
                if overwrite: 
                    self.xrdfs_client.rm(dest_file)
                    self.call_xrdcp(src_file, dest_file)
                    logging.debug("Overwriting file %s", dest_file)
            else:
                self.call_xrdcp(src_file, dest_file)
                logging.debug("Copying file %s", dest_file)
            # at some point needs to try copyprocess 
            # status, _ = self.xrdfs_client.copy(src_file, dest_file, force=True)
            # if not status.ok:
            if remove:
                os.remove(src_file)
            else:
                continue

def checkx509():
    """Check if the X509 proxy and certificate directory are set."""
    proxy_position = os.environ.get("X509_USER_PROXY", default=None)
    if proxy_position is None:
        raise SystemError("Proxy not found. Immediately check proxy!")
    logging.info(f"Proxy at: {proxy_position}.")
    proxy_directory = os.environ.get("X509_CERT_DIR", default=None)
    if proxy_directory is None:
        print(f"Certificate directory not set!")
    else:
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

def print_format_directory_structure(dir_structure) -> None:
    """
    Format a directory structure dictionary into a readable string and print it
    
    Args:
        dir_structure: Dictionary with structure {year: {groupname: {"lastUpdated": timestamp, "size": Megabytes}}}
    
    """
    output = []
    
    for year in sorted(dir_structure.keys()):
        output.append(f"\n{year}:")
        for group, details in sorted(dir_structure[year].items()):
            # Convert timestamp to human readable format
            last_updated = datetime.fromtimestamp(details["lastUpdated"]).strftime("%Y-%m-%d %H:%M:%S")
            # Format size to 2 decimal places
            size = f"{details['size']:.2f}"
            
            output.append(f"  └── {group}")
            output.append(f"      ├── Last Updated: {last_updated}")
            output.append(f"      └── Size: {size} MB")
    
    print("\n".join(output))


if __name__ == "__main__":
    file_helper = FileSysHelper()
    arg = sys.argv[1]
    display_directory_stats(file_helper.query_directory_structure(arg))