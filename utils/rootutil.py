import ROOT

def check_corrupt(file_path):
    try:
        file = ROOT.TFile.Open(file_path, "READ")
        if file.IsZombie() or file.TestBit(ROOT.TFile.kRecovered) or not file.IsOpen():
            raise Exception("File is corrupted or truncated")
        file.Close()
    except Exception as e:
        print(f"File {file_path} might be truncated or corrupted. Error: {e}")
