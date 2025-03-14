import logging, json, re, os
import pandas as pd

from src.analysis.objutil import Object
from src.utils.filesysutil import FileSysHelper, pjoin, pbase, pdir
from src.utils.datautil import extract_leaf_values, CutflowProcessor, DataSetUtil
from src.utils.rootutil import RootFileHandler

class PostProcessor():
    """Class for loading and hadding data from skims/predefined selections produced directly by Processor.
    
    Attributes
    - `cfg`: a configuration dictionary-like object for post processing. Must contain the following entries: INPUTDIR, LOCALOUTPUT, METADATA
    - `inputdir`: the directory where the input files to be post-processed are stored
    - `meta_dict`: the metadata dictionary for all datasets. {Groupname: {Datasetname: {metadata}}}
    - `groups`: list of group names to process. If not provided, will grep from the input directory.
    - `years`: list of years to process. If not provided, will grep from the input directory.
    - `luminosity`: per-year luminosity info in dictionary, in pb^-1"""
    def __init__(self, ppcfg, luminosity, groups=None, years=None) -> None:
        """Parameters
        - `ppcfg`: a configuration dictionary for post processing of datasets"""
        self.cfg = ppcfg
        self.lumi = luminosity
        self.inputdir = ppcfg['INPUTDIR']
        self.tempdir = ppcfg['LOCALOUTPUT']
        self.transferP = ppcfg.get("TRANSFERPATH", None)
        self.__will_trsf = False if self.transferP is None else True

        if self.__will_trsf: FileSysHelper.checkpath(self.transferP, createdir=True, raiseError=False)
        FileSysHelper.checkpath(self.inputdir, createdir=False, raiseError=True)
        FileSysHelper.checkpath(self.tempdir, createdir=True, raiseError=False)
    
        self.meta_dict = {}
            
        if years is None:
            self.years = [pbase(subdir) for subdir in FileSysHelper.glob_subdirs(self.inputdir, full_path=False)]
        else:
            self.years = years
            for year in self.years:
                FileSysHelper.checkpath(pjoin(self.inputdir, year), createdir=False, raiseError=True)
        
        for year in self.years:
            with open(pjoin(ppcfg['METADATA'], f"{year}.json"), 'r') as f:
                self.meta_dict[year] = json.load(f)
            if self.__will_trsf: FileSysHelper.checkpath(pjoin(self.transferP, year), createdir=True, raiseError=False)

        if groups is None:
            self.groups = self.__generate_groups
        else:
            self.groups = lambda year: groups
    
    def __del__(self):
        FileSysHelper.remove_emptydir(self.tempdir)
    
    def __generate_groups(self, year):
        """Generate the groups to process based on the input directory."""
        return [pbase(subdir) for subdir in FileSysHelper.glob_subdirs(pjoin(self.inputdir, year), full_path=False)]

    def __iterate_meta(self, callback) -> dict:
        """Iterate over datasets in a group over all groups and apply the callback function. Transfer any files in the local output directory to the transfer path (if set). 
        Collect the returns of the callback function in a dictionary.
        
        Parameters
        - `callback`: the function to apply to each dataset. Expects output in the temp directory. Callback has the signature (dsname, dtdir, outdir)."""
        results = {}
        for year in self.years:
            results[year] = {}
            meta = self.meta_dict[year]
            for group in self.groups(year):
                results[year][group] = {}
                datasets = meta[group]
                for _, dsitems in datasets.items():
                    dsname = dsitems['shortname']
                    dtdir = pjoin(self.inputdir, year, group)
                    outdir = pjoin(self.tempdir, year, group)
                    FileSysHelper.checkpath(outdir, createdir=True)
                    if not FileSysHelper.checkpath(dtdir, createdir=False): continue
                    results[year][group][dsname] = callback(dsname, dtdir, outdir)
                    if self.__will_trsf:
                        transferP = f"{self.transferP}/{year}/{group}"
                        FileSysHelper.transfer_files(outdir, transferP, remove=True, overwrite=True)
        return results
    
    def __update_meta(self):
        for year in self.years:
            with open(pjoin(pdir(self.cfg['METADATA']), 'weightedMC', f"{year}.json"), 'r') as f:
                self.meta_dict[year] = json.load(f)

    def __call__(self, output_type=None, outputdir=None):
        """Hadd the root/csv files of the datasets and save to the output directory."""
        if output_type is None:
            output_type = self.cfg.get("OUTTYPE", 'root')
        if outputdir is None:
            outputdir = self.tempdir if not self.__will_trsf else self.transferP
        
        FileSysHelper.checkpath(outputdir)

        self.hadd_cfs()
        if output_type == 'root': 
            self.hadd_roots()
            self.update_wgt_info(outputdir)
        elif output_type == 'csv': self.hadd_csvouts()
        else: raise TypeError("Invalid output type. Please choose either 'root' or 'csv'.")
        
    def get_yield(self):
        """Get the yield for the resolved cutflow tables. Save to LOCALOUTPUT."""
        self.__update_meta()
        regroup_dict = {"Others": ['WJets', 'WZ', 'WW', 'WWW', 'ZZZ', 'WZZ'], 'HH': ['ggF']}
        signals = ['HH', 'ZH', 'ZZ']
        inputdir = self.transferP if self.transferP is not None else self.inputdir
        _, combined_dict = self.merge_cf(inputdir=inputdir, outputdir=self.tempdir)
        for year, combined in combined_dict.items():
            self.present_yield(combined, signals, pjoin(self.tempdir, year), regroup_dict)
            print(f"Yield results are outputted in {pjoin(self.tempdir, year)}")
    
    def update_wgt_info(self, outputdir) -> None:
        """Output the weight information based on per-year per-dataset xsec to a json file."""
        new_meta_dir = pjoin(pdir(self.cfg['METADATA']), 'weightedMC')
        FileSysHelper.checkpath(new_meta_dir)
        for year in self.years:
            self.meta_dict[year] = PostProcessor.calc_wgt(pjoin(outputdir, year), self.meta_dict[year],
                                pjoin(new_meta_dir, f'{year}.json'), self.groups(year))
    
    def check_roots(self) -> None:
        """Check if the root files are corrupted by checking if the required keys are present. Save the corrupted files (full path with xrdfs prefix) to a text file.
        
        Parameters
        - `rq_keys`: list of required keys to check for in the root files"""
        corrupted ={}
        for year in self.years:
            corrupted[year] = {}
            for group in self.groups(year):
                ori_samples = pjoin(self.cfg['DATA_DIR'], 'preprocessed', f"{group}_{year}.json.gz")
                dt_dir = pjoin(self.inputdir, year, group)
                results = DataSetUtil.validate_file_pairs(ori_samples, dt_dir, dt_dir, n_workers=2)
                corrupted[year][group] = results
        if corrupted:
            logging.warning(f"Corrupted files found: {corrupted}")
            with open('all_corrupted.json', 'w') as f:
                json.dump(corrupted, f)
            logging.error(f"Saved corrupted files to all_corrupted.json")
    
    def clean_roots(self) -> None:
        """Delete the corrupted files in the filelist."""
        for year in self.years:
            for group in self.groups(year):
                if os.path.exists(f'{group}_corrupted_files.txt'):
                    self.delete_corrupted(f'{group}_corrupted_files.txt')
        if os.path.exists('all_corrupted.txt'):
            self.delete_corrupted('all_corrupted.txt')
    
    def hadd_roots(self) -> str:
        """Hadd root files of datasets into appropriate size based on settings"""
        def process_ds(dsname, dtdir, outdir):
            root_files = FileSysHelper.glob_files(dtdir, f'{dsname}*.root', add_prefix=True)
            batch_size = self.cfg.get("BATCHSIZE", 200)
            corrupted = set()
            for i in range(0, len(root_files), batch_size):
                batch_files = root_files[i:i+batch_size]
                outname = pjoin(outdir, f"{dsname}_{i//batch_size+1}.root") 
                try:
                    new_corrupt = call_hadd(outname, batch_files)
                    if new_corrupt is not None:
                        print(f"Error merging filename batch {i}")
                        corrupted |= new_corrupt
                except Exception as e:
                    print(f"Hadding encountered error {e}")
                    print(batch_files)
            return list(corrupted)
        
        all_corrupted = extract_leaf_values(self.__iterate_meta(process_ds))
        all_corrupted_list = [fn for sub in all_corrupted for fn in sub]
        if all_corrupted_list:
            with open('all_corrupted.txt', 'w') as f:
                f.write('\n'.join(all_corrupted))

    def hadd_csvouts(self) -> None:
        """Hadd csv output files of datasets into one csv file"""
        def process_csv(dsname, dtdir, outdir):
            concat = lambda dfs: pd.concat(dfs, axis=0)
            try:
                df = load_csvs(dtdir, f'{dsname}*output*csv', func=concat)
                df.to_csv(pjoin(outdir, f"{dsname}_out.csv"))
            except Exception as e:
                print(f"Error loading csv files for {dsname}: {e}")
        
        self.__iterate_meta(process_csv)
        
    def hadd_cfs(self):
        """Hadd cutflow table output from processor. Output a total cutflow for the group with all the sub datasets."""
        def process_cf(dsname, dtdir, outdir):
            print(f"Dealing with {dsname} cutflow hadding now ...............................")
            try:
                df = CutflowProcessor.merge_cutflows(inputdir=dtdir, dsname=dsname, output=False)
                df.to_csv(pjoin(outdir, f"{dsname}_cutflow.csv"))
                return df
            except Exception as e:
                print(f"Error combining cutflow tables for {dsname}: {e}")
        
        group_dfs = self.__iterate_meta(process_cf)
        for year, grouped in group_dfs.items():
            for group, nested in grouped.items():
                total_df = pd.concat(nested.values(), axis=1)
                total_df.to_csv(pjoin(self.tempdir, f"{group}_{year}_cf.csv"))
                if self.__will_trsf:
                    transferP = f"{self.transferP}/{year}/{group}"
                    FileSysHelper.transfer_files(self.tempdir, transferP, filepattern='*csv', remove=True, overwrite=True)
                
    def merge_cf(self, inputdir, outputdir, extra_kwd=''):
        """Merges and weights all cutflow tables across years and groups.
        
        Creates three output files per year:
        - allDatasetCutflow.csv: Raw and weighted events for all datasets
        - ResolvedWgtOnly.csv: Only weighted events
        - ResolvedEffOnly.csv: Selection efficiencies
        
        Returns
        -------
        tuple[dict, dict]
            (Weighted cutflows per year, Combined weights per year)
        """
        resolved_wgted = {}
        combined_wgted = {}

        self.__update_meta()

        for year in self.years:
            # Setup output directory
            year_outdir = pjoin(outputdir, year)
            FileSysHelper.checkpath(year_outdir, createdir=True)
            
            resolved_list = []
            combined_list = []
            
            # Process each group
            for group in self.groups(year):
                if not FileSysHelper.checkpath(pjoin(inputdir, year, group)):
                    continue
                    
                resolved, combined = self.process_group_cutflow(
                    group,
                    self.meta_dict[year],
                    pjoin(inputdir, year),
                    self.lumi[year],
                    extra_kwd
                )
                resolved_list.append(resolved)
                combined_list.append(combined)

            # Combine results
            resolved_all = pd.concat(resolved_list, axis=1)
            resolved_all.to_csv(pjoin(year_outdir, "allDatasetCutflow.csv"))
            
            # Extract weighted events
            wgt_resolved = resolved_all.filter(like='wgt', axis=1)
            wgt_resolved.columns = wgt_resolved.columns.str.replace('_wgt$', '', regex=True)
            wgt_resolved.to_csv(pjoin(year_outdir, "ResolvedWgtOnly.csv"))
            
            # Calculate efficiencies
            eff_df = CutflowProcessor.calculate_efficiency(wgt_resolved)
            eff_df.filter(like='eff', axis=1).to_csv(
                pjoin(year_outdir, "ResolvedEffOnly.csv")
            )
            
            # Store results
            resolved_wgted[year] = wgt_resolved
            combined_wgted[year] = pd.concat(combined_list, axis=1)
        
        return resolved_wgted, combined_wgted 
    
    @staticmethod
    def present_yield(wgt_resolved, signals, outputdir, regroup_dict=None) -> pd.DataFrame:
        """Present the yield dataframe with grouped datasets. Regroup if necessary.
        
        Parameters
        - `signals`: list of signal group names
        - `regroup_dict`: dictionary of regrouping keywords. Passed into `PostProcessor.categorize`.
        """
        if regroup_dict is not None:
            wgt_resolved = CutflowProcessor.categorize_processes(wgt_resolved, regroup_dict)
        
        yield_df = CutflowProcessor.calculate_yield(wgt_resolved, signals)
        yield_df.to_csv(pjoin(outputdir, 'scaledyield.csv'))
        
        return yield_df
    
    @staticmethod
    def calc_wgt(datasrcpath, meta_dict, dict_outpath, groups) -> dict:
        """Calculate the sum of `Generator_weight` per each dataset and save to a json file with provided metadata.
        
        Parameters
        - `datasrcpath`: path to the output directory (base level)
        - `meta_dict`: metadata dictionary for all datasets. {group: {dataset: {metadata}}}"""
        new_meta = {}
        if os.path.exists(dict_outpath):
            with open(dict_outpath, 'r') as f:
                new_meta = json.load(f)

        for group in groups:
            print(f"Globbing from {pjoin(datasrcpath, group)}")
            resolved_df = pd.read_csv(FileSysHelper.glob_files(pjoin(datasrcpath, group), f'{group}*cf.csv')[0], index_col=0) 
            if not(group in new_meta):
                new_meta[group] = meta_dict[group].copy()
            for ds, dsitems in meta_dict[group].items(): 
                if not (ds in new_meta[group]):
                    new_meta[group][ds] = meta_dict[group][ds].copy()
                if not resolved_df.filter(like=dsitems['shortname']).empty:
                    nwgt = resolved_df.filter(like=dsitems['shortname']).filter(like='wgt').iloc[0,0]
                    new_meta[group][ds]['nwgt'] = nwgt
                else:
                    del new_meta[group][ds]
        
        with open(pjoin(dict_outpath), 'w') as f:
            json.dump(new_meta, f)
        
        return new_meta

    @staticmethod
    def process_group_cutflow(group, meta_dict, inputdir, luminosity, extra_kwd=''):
        """Process cutflow tables for a physics group with appropriate scaling.
        
        Parameters
        ----------
        group : str
            Name of the physics process group
        meta_dict : dict
            Metadata dictionary for the group
        inputdir : str
            Input directory containing cutflow files
        luminosity : float
            Luminosity in pb^-1
        extra_kwd : str, optional
            Additional keyword for file matching
            
        Returns
        -------
        tuple[pd.DataFrame, pd.Series]
            (Scaled cutflow dataframe, Combined group weights)
        """
        # Load cutflow
        cutflow_df = pd.read_csv(
            FileSysHelper.glob_files(
                pjoin(inputdir, group), 
                f'{group}*{extra_kwd}*cf.csv'
            )[0],
            index_col=0
        )
        
        # Apply physics scaling
        scaled_df, combined = CutflowProcessor.apply_physics_scale(
            cutflow_df,
            meta_dict[group],
            luminosity
        )
        
        # Set group name for combined weights
        combined.name = group
        
        return scaled_df, combined

    @staticmethod
    def delete_corrupted(results_dict):
        """Delete the corrupted files in the filelist."""
        with open(filelist_path, 'r') as f:
            filelist = f.read().splitlines()
        
        dirname = pdir(filelist[0]).replace('root://cmseos.fnal.gov/', '')
        
        filelist = [file.rsplit('/', 1)[-1] for file in filelist]
        filelist = [re.sub(r'-part\d+\.root', '*', delfile) for delfile in filelist]
        filelist = list(set(filelist))

        for file in filelist: FileSysHelper.remove_files(dirname, file)
        



