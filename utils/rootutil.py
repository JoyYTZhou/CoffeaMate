import uproot, logging, subprocess, re, sys
import awkward as ak
from src.utils.filesysutil import FileSysHelper
from rich import print as rprint

class RootFileHandler:
    @staticmethod
    def check_root_file(file_path) -> bool:
        """Check if a ROOT file is empty or corrupt.
        - file_path: path to the ROOT file

        Returns:
        - True if file is empty or corrupt, False otherwise
        """
        MIN_SIZE_BYTES = 1024 * 2
        try:
            file_size = FileSysHelper.get_file_size(file_path)
            if file_size < MIN_SIZE_BYTES:
                logging.warning(f"File {file_path} is too small (< {MIN_SIZE_BYTES} bytes)")
                return True

            with uproot.open(file_path) as file:
                # Check if file can be opened
                if file is None:
                    logging.error(f"File {file_path} cannot be opened")
                    return True
                keys = file.keys()
                if len(keys) == 0:
                    logging.warning(f"File {file_path} has no keys")
                    return True

                for key in keys:
                    try:
                        data = file[key]
                        if hasattr(data, 'num_entries') and data.num_entries == 0:
                            continue
                        elif hasattr(data, '__len__') and len(data) == 0:
                            continue
                        else:
                            return False  # Found non-empty, valid data
                    except Exception as e:
                        logging.debug(f"Could not read key {key}: {e}")
                        continue

                logging.warning(f"File {file_path} contains keys but all are empty")
                return True
        except Exception as e:
            logging.error(f"Error checking file {file_path}: {e}")
            return True

    @staticmethod
    def call_hadd(output_file, input_files) -> tuple[int, set]:
        """Merge ROOT files using hadd.

        Parameters
        - `output_file`: path to the output file
        - `input_files`: list of paths to the input files
        
        Return
        """
        command = ['hadd', '-f2 -O -k', output_file] + input_files
        result = subprocess.run(command, capture_output=True, text=True)
        unique_filenames = None
        if result.returncode == 0:
            print(f"Merged files into {output_file}")
        else:
            print(f"Error merging files: {result.stderr}")    
            filenames = re.findall(r"root://[^\s]*\.root", result.stderr)
            unique_filenames = set(filenames)
        return unique_filenames

    @staticmethod
    def find_branches(file_path, object_list, tree_name, extra=[]) -> list:
        """Return a list of branches for objects in object_list

        Paremters
        - `file_path`: path to the root file
        - `object_list`: list of objects to find branches for
        - `tree_name`: name of the tree in the root file
        - `extra`: list of extra branches to include

        Returns
        - list of branches
        """
        file = uproot.open(file_path)
        tree = file[tree_name]
        branch_names = tree.keys()
        branches = []
        for object in object_list:
            branches.extend([name for name in branch_names if name.startswith(object)])
        if extra != []:
            branches.extend([name for name in extra if name in branch_names])
        return branches
    
    @staticmethod
    def get_compression(**kwargs):
        """Returns the compression algorithm to use for writing root files."""
        compression = kwargs.pop('compression', None)
        compression_level = kwargs.pop('compression_level', 1)

        if compression in ("LZMA", "lzma"):
            compression_code = uproot.const.kLZMA
        elif compression in ("ZLIB", "zlib"):
            compression_code = uproot.const.kZLIB
        elif compression in ("LZ4", "lz4"):
            compression_code = uproot.const.kLZ4
        elif compression in ("ZSTD", "zstd"):
            compression_code = uproot.const.kZSTD
        elif compression is None:
            raise UserWarning("Not sure if this option is supported, should be...")
        else:
            msg = f"unrecognized compression algorithm: {compression}. Only ZLIB, LZMA, LZ4, and ZSTD are accepted."
            raise ValueError(msg)
        
        if compression is not None: 
            compression = uproot.compression.Compression.from_code_pair(compression_code, compression_level)

        return compression

    @staticmethod
    def print_total_wgt(file_path, tree_name='Events', branch_name='Generator_weight') -> float:
        """Print the total weight of the given branch in the given tree."""
        total_weight = ak.sum(uproot.open(file_path)[tree_name][branch_name].array())
        rprint(f"Total weight for {branch_name} in {tree_name}: [cyan]{total_weight}[/cyan]")

        return total_weight

    # @staticmethod
    # def write_obj(writable, filelist, objnames, extra=[]) -> None:
    #     """Writes the selected, concated objects to root files.
    #     Parameters:
    #     - `writable`: the uproot.writable directory
    #     - `filelist`: list of root files to extract info from
    #     - `objnames`: list of objects to load. Required to be entered in the selection config file.
    #     - `extra`: list of extra branches to save"""

    #     all_names = objnames + extra
    #     all_data = {name: [] for name in objnames}
    #     all_data['extra'] = {name: [] for name in extra}
    #     for file in filelist:
    #         evts = load_fields(file)
    #         print(f"events loaded for file {file}")
    #         for name in all_names:
    #             if name in objnames:
    #                 obj = Object(evts, name)
    #                 zipped = obj.getzipped()
    #                 all_data[name].append(zipped)
    #             else:
    #                 all_data['extra'][name].append(evts[name])
    #     for name, arrlist in all_data.items():
    #         if name != 'extra':
    #             writable[name] = ak.concatenate(arrlist)
    #         else:
    #             writable['extra'] = {branchname: ak.concatenate(arrlist[branchname]) for branchname in arrlist.keys()}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rootutil.py <root_file> [tree_name]")
        sys.exit(1)

    root_file = sys.argv[1]
    tree_name = sys.argv[2] if len(sys.argv) > 2 else "Events"

    try:
        RootFileHandler.print_total_wgt(root_file, tree_name)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)