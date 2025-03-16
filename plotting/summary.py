import logging, json, re, os
import pandas as pd
from itertools import chain

from src.utils.filesysutil import FileSysHelper, pjoin, pbase, pdir
from src.utils.datautil import CutflowProcessor, DataSetUtil, DatasetIterator
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
        """Initialize the PostProcessor.
        Parameters
        ----------
        ppcfg : dict
            Configuration dictionary for post processing
        luminosity : dict
            Per-year luminosity info in pb^-1
        groups : list, optional
            List of group names to process. If None, will grep from input directory
        years : list, optional
            List of years to process. If None, will grep from input directory
        """
        self.cfg = ppcfg
        self.lumi = luminosity

        self._init_paths()

        self._init_years(years)

        self._init_metadata()

        self._init_groups(groups)

        self.dataset_iter = DatasetIterator(
            years=years,
            groups_func=self.groups,
            meta_dict=self.meta_dict,
            input_root_dir=self.inputdir,
            temp_root_dir=self.tempdir,
            transfer_root=self.transferP
        )

    def _init_paths(self):
        """Initialize and validate input/output paths."""
        self.inputdir = self.cfg['INPUTDIR']
        self.tempdir = self.cfg['LOCALOUTPUT']
        self.transferP = self.cfg.get("TRANSFERPATH", None)
        self.__will_trsf = self.transferP is not None
        
        logging.debug(f"Input directory: {self.inputdir}")
        logging.debug(f"Output directory: {self.tempdir}")
        if self.__will_trsf:
            logging.debug(f"Transfer directory: {self.transferP}")

        # Validate paths
        FileSysHelper.checkpath(self.inputdir, createdir=False, raiseError=True)
        FileSysHelper.checkpath(self.tempdir, createdir=True, raiseError=False)
        if self.__will_trsf:
            FileSysHelper.checkpath(self.transferP, createdir=True, raiseError=False)

    def _init_years(self, years):
        """Initialize processing years."""
        if years is None:
            self.years = [pbase(subdir) for subdir in FileSysHelper.glob_subdirs(self.inputdir, full_path=False)]
        else:
            self.years = years
            for year in self.years:
                FileSysHelper.checkpath(pjoin(self.inputdir, year), createdir=False, raiseError=True)

        if self.__will_trsf:
            for year in self.years:
                FileSysHelper.checkpath(pjoin(self.transferP, year), createdir=True, raiseError=False)

    def _init_metadata(self):
        """Load metadata for each year."""
        self.meta_dict = {}
        for year in self.years:
            with open(pjoin(self.cfg['METADATA'], f"{year}.json"), 'r') as f:
                    self.meta_dict[year] = json.load(f)

    def _init_groups(self, groups):
        """Initialize processing groups."""
        if groups is None:
            self.groups = self.__generate_groups
        else:
            self.groups = lambda year: groups
                
    def __del__(self):
        FileSysHelper.remove_emptydir(self.tempdir)
    
    def __generate_groups(self, year):
        """Generate the groups to process based on the input directory."""
        return [pbase(subdir) for subdir in FileSysHelper.glob_subdirs(pjoin(self.inputdir, year), full_path=False)]
    
    def __update_meta(self):
        for year in self.years:
            with open(pjoin(pdir(self.cfg['METADATA']), 'weightedMC', f"{year}.json"), 'r') as f:
                self.meta_dict[year] = json.load(f)
    
    def hadd_results(self, *args, **kwargs):
        """Hadd the results of the post processing. The method is to be implemented in the child class."""
        raise NotImplementedError("hadd_results method is not implemented in the child class.")

    def check_and_clean(self, *args, **kwargs):
        """Check the results of the post processing. The method is to be implemented in the child class."""
        raise NotImplementedError("check_and_clean method is not implemented in the child class.")
    
    def __call__(self, checkargs, haddargs):
        self.check_and_clean(*checkargs)
        self.hadd_results(*haddargs)

    # def __call__(self, output_type=None, outputdir=None):
    #     """Hadd the root/csv files of the datasets and save to the output directory."""
    #     if output_type is None:
    #         output_type = self.cfg.get("OUTTYPE", 'root')
    #     if outputdir is None:
    #         outputdir = self.tempdir if not self.__will_trsf else self.transferP
        
    #     FileSysHelper.checkpath(outputdir)

    #     self.hadd_cfs()
    #     if output_type == 'root': 
    #         self.hadd_roots()
    #         self.update_wgt_info(outputdir)
    #     elif output_type == 'csv': self.hadd_csvouts()
    #     else: raise TypeError("Invalid output type. Please choose either 'root' or 'csv'.")
        
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

    # def hadd_csvouts(self) -> None:
    #     """Hadd csv output files of datasets into one csv file"""
    #     def process_csv(dsname, dtdir, outdir):
    #         concat = lambda dfs: pd.concat(dfs, axis=0)
    #         try:
    #             df = load_csvs(dtdir, f'{dsname}*output*csv', func=concat)
    #             df.to_csv(pjoin(outdir, f"{dsname}_out.csv"))
    #         except Exception as e:
    #             print(f"Error loading csv files for {dsname}: {e}")
        
    #     self.__iterate_meta(process_csv)
                        
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


class PostSkimProcessor(PostProcessor):
    def __init__(self, ppcfg, luminosity, groups=None, years=None) -> None:
        super().__init__(ppcfg, luminosity, groups, years)
    
    def check_and_clean(self):
        self.__check_roots()
        self.__clean_roots()
    
    def check_results(self):
        self.__check_roots()
    
    def clean_results(self):
        self.__clean_roots()
    
    def hadd_results(self):
        self.__hadd_roots()
        self.__hadd_cutflows()
    
    def __clean_roots(self):
        """Delete the corrupted files in the filelist."""
        if os.path.exists('all_corrupted.json'):
            with open('all_corrupted.json', 'r') as f:
                all_corrupted = json.load(f)
            self.__delete_corrupted(all_corrupted)
    
    def __check_roots(self) -> dict:
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
        
        return corrupted

    def __hadd_roots(self) -> str:
        """Hadd root files of datasets into appropriate size based on settings"""
        def process_ds(dsname, dtdir, outdir):
            root_files = FileSysHelper.glob_files(dtdir, f'{dsname}*.root', add_prefix=True)
            batch_size = 100 
            corrupted = set()
            for i in range(0, len(root_files), batch_size):
                batch_files = root_files[i:i+batch_size]
                outname = pjoin(outdir, f"{dsname}_{i//batch_size+1}.root") 
                try:
                    new_corrupt = RootFileHandler.call_hadd(outname, batch_files)
                    if new_corrupt is not None:
                        print(f"Error merging filename batch {i}")
                        corrupted |= new_corrupt
                except Exception as e:
                    logging.error(f"Hadding {dsname} encountered error {e}")
                    logging.info(batch_files)
            return list(corrupted)
        
        results = DataSetUtil.extract_leaf_values(self.dataset_iter.process_datasets(process_ds))
        corrupted = list(chain(*results))
        
        if corrupted:
            with open('all_corrupted.txt', 'w') as f:
                f.write('\n'.join(corrupted))

    def __hadd_cutflows(self):
        """Hadd cutflow table output from processor. Output a total cutflow for the group with all the sub datasets."""
        def process_cf(dsname, dtdir, outdir):
            print(f"Dealing with {dsname} cutflow hadding now ...............................")
            try:
                df = CutflowProcessor.merge_cutflows(inputdir=dtdir, dsname=dsname, save=False)
                df.to_csv(pjoin(outdir, f"{dsname}_cutflow.csv"))
                return df
            except Exception as e:
                logging.error(f"Error combining cutflow tables for {dsname}: {e}")
                return None
        
        results = self.dataset_iter.process_datasets(process_cf)
        
        for year, grouped in results.items():
            for group, nested in grouped.items():
                valid_dfs = {k: v for k, v in nested.items() if v is not None}
                if valid_dfs:
                    total_df = pd.concat(valid_dfs.values(), axis=1)
                    total_df.to_csv(pjoin(self.tempdir, f"{group}_{year}_cf.csv"))
                    if self.__will_trsf:
                        transferP = f"{self.transferP}/{year}/{group}"
                        FileSysHelper.transfer_files(self.tempdir, transferP, filepattern='*csv', remove=True, overwrite=True)

    def __delete_corrupted(self, results_dict):
        """Delete the corrupted files in the filelist."""
        for year in results_dict:
            for group in results_dict[year]:
                all_mismatched = results_dict[year][group]['mismatched_events']
                root_files = [d.get('root') for d in all_mismatched]
                logging.debug(f"Deleting corrupted files: {root_files}")
                FileSysHelper.remove_filelist(root_files)
                csv_files = [d.get('csv') for d in all_mismatched]
                logging.debug(f"Deleting corrupted files: {csv_files}")
                FileSysHelper.remove_filelist(csv_files)
        
    # def __get_total_nwgt_events(self):
    #     """Calculate the sum of `Generator_weight` per each dataset and save to a json file with provided metadata.
        
    #     Parameters
    #     - `datasrcpath`: path to the output directory (base level)
    #     - `meta_dict`: metadata dictionary for all datasets. {group: {dataset: {metadata}}}"""
    #     new_meta_outpath = pjoin(self.cfg['DATA_DIR'], 'weightedMC')
    #     FileSysHelper.checkpath(new_meta_outpath, createdir=True)
        
        # def get_nwgt_per_group(group):
        #     resolved_df = pd.read_csv(FileSysHelper.glob_files(pjoin(datasrcpath, group), f'{group}*cf.csv')[0], index_col=0)
        #     for ds, dsitems in meta_dict[group].items():
        #         if not resolved_df.filter(like=dsitems['shortname']).empty:
        #             nwgt = resolved_df.filter(like=dsitems['shortname']).filter(like='wgt').iloc[0,0]
        #             meta_dict[group][ds]['nwgt'] = nwgt
        #         else:
        #             del meta_dict[group][ds]
        
        #     if os.path.exists(dict_outpath):
        #         with open(dict_outpath, 'r') as f:
        #             new_meta = json.load(f)

        #     for group in groups:
        #         print(f"Globbing from {pjoin(datasrcpath, group)}")
        #         resolved_df = pd.read_csv(FileSysHelper.glob_files(pjoin(datasrcpath, group), f'{group}*cf.csv')[0], index_col=0) 
        #         if not(group in new_meta):
        #             new_meta[group] = meta_dict[group].copy()
        #         for ds, dsitems in meta_dict[group].items(): 
        #             if not (ds in new_meta[group]):
        #                 new_meta[group][ds] = meta_dict[group][ds].copy()
        #             if not resolved_df.filter(like=dsitems['shortname']).empty:
        #                 nwgt = resolved_df.filter(like=dsitems['shortname']).filter(like='wgt').iloc[0,0]
        #                 new_meta[group][ds]['nwgt'] = nwgt
        #             else:
        #                 del new_meta[group][ds]
            
        #     with open(pjoin(dict_outpath), 'w') as f:
        #         json.dump(new_meta, f)
            
        #     return new_meta