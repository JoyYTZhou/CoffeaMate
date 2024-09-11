import unittest, os, glob

from config.selectionconfig import runsetting as rs
from src.analysis.spawnjobs import JobLoader
import json

pjoin = os.path.join

class TestLoader(unittest.TestCase):
    def setUp(self): 
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        analysis_path = os.path.dirname(curr_dir)
        self.datapath = pjoin(analysis_path, "data/preprocessed")
        self.outpath = pjoin(rs.OUTPUTDIR_PATH, "testout")
        self.jl = JobLoader(jobpath=self.outpath, datapath=self.datapath)
    
    def test_init(self):
        self.assertTrue(os.path.exists(self.outpath), f"Directory {self.datapath} does not exist!")
    
    def tearDown(self):
        files = glob.glob(pjoin(self.outpath, "*.json"))
        for f in files: os.remove(f)
    
    def test_prep_jobs(self):
        inputpath = pjoin(self.datapath, "ZZ.json.gz")
        returned = self.jl.prepjobs(inputpath)

        if returned:
            files = glob.glob(pjoin(self.outpath, "*.json"))
            self.assertTrue(len(files) > 0, "No files were written!")
            with open(files[0], 'r') as f:
                data = json.load(f)
                self.assertIn("files", data, 'Double check file generation in JobLoader.prepjobs')
                self.assertIn("metadata", data, "Double check metadata generation in JobLoader.prepjobs")
                self.assertEqual(len(data["files"]), 15, "Double check file division in JobLoader.prepjobs")

if __name__ == '__main__':
    unittest.main()

    