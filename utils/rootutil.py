import ROOT
import os, uproot, logging

def check_corrupt(file_path):
    try:
        file = ROOT.TFile.Open(file_path, "READ")
        if file.IsZombie() or file.TestBit(ROOT.TFile.kRecovered) or not file.IsOpen():
            raise Exception("File is corrupted or truncated")
        file.Close()
    except Exception as e:
        print(f"File {file_path} might be truncated or corrupted. Error: {e}")

def check_empty(file_path) -> bool:
    """Check if a ROOT file is empty."""
    # First check file size - if it's larger than 1KB, consider it non-empty
    MIN_SIZE_BYTES = 1024  # 1KB threshold, adjust as needed

    try:
        file_size = os.path.getsize(file_path)
        if file_size > MIN_SIZE_BYTES:
            return False

        # If file is very small, do detailed check
        with uproot.open(file_path) as file:
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
                        return False  # Found non-empty data
                except Exception as e:
                    logging.debug(f"Could not read key {key}: {e}")
                    continue

            logging.warning(f"File {file_path} contains keys but all are empty")
            return True

    except Exception as e:
        logging.error(f"Error checking file {file_path}: {e}")
        return True
