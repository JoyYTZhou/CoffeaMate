import logging, json, os
from datetime import datetime
import numpy as np
import pandas as pd
from itertools import chain

from src.utils.filesysutil import FileSysHelper, pjoin, pbase, pdir
from src.utils.datautil import CutflowProcessor, DataSetUtil, DatasetIterator, DataLoader
from src.utils.rootutil import RootFileHandler
from src.utils.displayutil import create_table, print_dataframe_rich

luminosity = {"2022PreEE": 41.5/2 * 1000, "2022PostEE": 41.5 * 1000/2, "2023Summer": 32.7 * 1000}

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

        self._init_iter()

    def _init_paths(self):
        """Initialize and validate input/output paths."""
        self.inputdir = self.cfg['INPUTDIR']
        self.tempdir = self.cfg['LOCALOUTPUT']
        self.transferP = self.cfg.get("TRANSFERPATH", None)
        self._will_trsf = self.transferP is not None
        
        logging.debug(f"Input directory: {self.inputdir}")
        logging.debug(f"Output directory: {self.tempdir}")
        if self._will_trsf:
            logging.debug(f"Transfer directory: {self.transferP}")

        # Validate paths
        FileSysHelper.checkpath(self.inputdir, createdir=False, raiseError=True)
        FileSysHelper.checkpath(self.tempdir, createdir=True, raiseError=False)
        if self._will_trsf:
            FileSysHelper.checkpath(self.transferP, createdir=True, raiseError=False)

    def _init_years(self, years):
        """Initialize processing years."""
        if years is None:
            self.years = [pbase(subdir) for subdir in FileSysHelper.glob_subdirs(self.inputdir, full_path=False)]
        else:
            self.years = years
            for year in self.years:
                FileSysHelper.checkpath(pjoin(self.inputdir, year), createdir=False, raiseError=True)

        if self._will_trsf:
            for year in self.years:
                FileSysHelper.checkpath(pjoin(self.transferP, year), createdir=True, raiseError=False)
        
        logging.debug(f"Processing years: {self.years}")

        logging.debug(f"Processing years: {self.years}")

    def _init_metadata(self):
        """Load metadata for each year."""
        self.meta_dict = {}
        query_dir = pjoin(self.cfg['DATA_DIR'], 'weightedMC' if self.cfg['IS_MC'] else 'availableData')
        logging.debug(f"Getting metadata from {query_dir}")
        for year in self.years:
            logging.debug(f"Getting metadata for {year}.")
            if os.path.exists(pjoin(query_dir, f"{year}.json")):
                with open(pjoin(query_dir, f"{year}.json"), 'r') as f:
                    self.meta_dict[year] = json.load(f)
        
    def _init_groups(self, groups):
        """Initialize processing groups.

        If groups is provided, only returns groups that exist in the year's directory.
        If groups is None, returns all groups found in the year's directory.
        """
        if groups is None:
            self.groups = lambda year: [pbase(subdir) for subdir in FileSysHelper.glob_subdirs(pjoin(self.inputdir, year), full_path=False)]
        else:
            self.groups = lambda year: [group for group in groups
                                      if FileSysHelper.checkpath(pjoin(self.inputdir, year, group), createdir=False, raiseError=False)]

    def _init_iter(self):
        """Initialize the dataset iterator for processing."""
        self.dataset_iter = DatasetIterator(
            years=self.years,
            groups_func=self.groups,
            meta_dict=self.meta_dict,
            input_root_dir=self.inputdir,
            temp_root_dir=self.tempdir,
            transfer_root=self.transferP
        ) 
        
    def __del__(self):
        FileSysHelper.remove_emptydir(self.tempdir)
    
    def __update_meta(self):
        for year in self.years:
            with open(pjoin(self.cfg['DATA_DIR'], 'weightedMC', f"{year}.json"), 'r') as f:
                self.meta_dict[year] = json.load(f)
        logging.info(f"Using metadata from {pjoin(self.cfg['DATA_DIR'], 'weightedMC', f'{year}.json')}")
        self._init_iter()
    
    def hadd_results(self, *args, **kwargs):
        """Hadd the results of the post processing. The method is to be implemented in the child class."""
        raise NotImplementedError("hadd_results method is not implemented in the child class.")

    def check_and_clean(self, *args, **kwargs):
        """Check the results of the post processing. The method is to be implemented in the child class."""
        raise NotImplementedError("check_and_clean method is not implemented in the child class.")
    
    def __call__(self, checkargs, haddargs):
        self.check_and_clean(*checkargs)
        self.hadd_results(*haddargs)
        
    def get_yield(self, extra_kwd=''):
        """Get the yield for the resolved cutflow tables. All the weights incorporate xsec * lumi.
        Save to LOCALOUTPUT."""
        self.__update_meta()
        regroup_dict = {"Others": ['WJets', 'WZ', 'WW', 'WWW', 'ZZZ', 'WZZ'], 'HH': ['ggF']}
        signals = ['HH', 'ZH', 'ZZ']
        inputdir = self.transferP if self.transferP is not None else self.inputdir
        resolved_dict, combined_dict = self.merge_cf(inputdir=inputdir, outputdir=self.tempdir, extra_kwd=extra_kwd)
        for year, combined in combined_dict.items():
            self.present_yield(combined, signals, pjoin(self.tempdir, year), regroup_dict)
            logging.info(f"Yield results are outputted in {pjoin(self.tempdir, year)}")
            print_dataframe_rich(combined, title=f"Yield for {year}")

        FileSysHelper.checkpath(pjoin(self.tempdir, 'allYears'))
        _, combined_all = self.combine_merge_cf_results(resolved_dict, combined_dict, pjoin(self.tempdir, 'allYears'))
        self.present_yield(combined_all, signals, pjoin(self.tempdir, 'allYears'), regroup_dict)
        logging.info(f"Yield results are outputted in {pjoin(self.tempdir, 'allYears')}")
        print_dataframe_rich(combined_all, title="Yield for all years")
    
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
        """Merges and weights all cutflow tables across years and groups. All the weights incorporate xsec * lumi.
        
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
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                print_dataframe_rich(resolved_all, title=f"Resolved Cutflow for {year}")
            resolved_all.to_csv(pjoin(year_outdir, "allDatasetCutflow.csv"))
            
            # Extract weighted events
            wgt_resolved = resolved_all.filter(like='wgt', axis=1)
            wgt_resolved.columns = wgt_resolved.columns.str.replace('_wgt$', '', regex=True)
            wgt_resolved.to_csv(pjoin(year_outdir, "ResolvedWgtOnly.csv"))
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                print_dataframe_rich(wgt_resolved, title=f"Resolved Weighted Cutflow for {year}")
            
            # Calculate efficiencies
            eff_df = CutflowProcessor.calculate_efficiency(wgt_resolved)
            eff_df.filter(like='eff', axis=1).to_csv(
                pjoin(year_outdir, "ResolvedEffOnly.csv")
            )
            
            # Store results
            resolved_wgted[year] = wgt_resolved
            combined_wgted[year] = pd.concat(combined_list, axis=1)
        
        return resolved_wgted, combined_wgted 
    
    def combine_merge_cf_results(self, resolved_wgted, combined_wgted, outputdir):
        """Combine merge_cf results across all years.

        Parameters
        ----------
        resolved_wgted : dict
            Weighted cutflows per year from merge_cf
        combined_wgted : dict
            Combined weights per year from merge_cf
        outputdir : str
            Output directory for combined results

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (weights resolved by channels, Combined weights)
        """
        # Add weighted cutflows across years
        combined_wgt_resolved = sum(resolved_wgted.values())
        combined_wgt_resolved.to_csv(pjoin(outputdir, "CombinedResolvedWgtOnly.csv"))

        # Add weights across years
        combined_total_wgt = sum(combined_wgted.values())
        combined_total_wgt.to_csv(pjoin(outputdir, "CombinedWeights.csv"))

        # Calculate combined efficiencies
        combined_eff = CutflowProcessor.calculate_efficiency(combined_wgt_resolved)
        combined_eff.filter(like='eff', axis=1).to_csv(pjoin(outputdir, "CombinedResolvedEffOnly.csv"))

        return combined_wgt_resolved, combined_total_wgt

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
        """Process cutflow tables for a physics group with appropriate scaling (lumi * xsec)
        
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
        # First read the CSV without forcing dtypes
        cutflow_df = pd.read_csv(
            FileSysHelper.glob_files(
                pjoin(inputdir, group), 
                f'{group}*{extra_kwd}*cf.csv',
            )[0],
            index_col=0,
            header=0
        )

        # Convert numeric columns to float64, leaving non-numeric columns as is
        numeric_columns = cutflow_df.select_dtypes(include=['int64', 'float64']).columns
        cutflow_df[numeric_columns] = cutflow_df[numeric_columns].astype(np.float64)
        
        # Apply physics scaling
        scaled_df, combined = CutflowProcessor.apply_physics_scale(
            cutflow_df,
            meta_dict[group],
            luminosity
        )
        
        # Set group name for combined weights
        combined.name = group
        
        return scaled_df, combined
    
    def _hadd_cutflows(self):
        """Hadd cutflow table output from processor. Output a total cutflow for the group with all the sub datasets."""
        def process_cf(dsname, dtdir, outdir):
            logging.info(f"Dealing with {dsname} cutflow hadding now ...............................")
            try:
                df = CutflowProcessor.merge_cutflows(inputdir=dtdir, dataset_name=dsname, save=True, outpath=pjoin(outdir, f"{dsname}_cutflow.csv"))
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
                    if self._will_trsf:
                        transferP = f"{self.transferP}/{year}/{group}"
                        FileSysHelper.transfer_files(self.tempdir, transferP, filepattern='*csv', remove=True, overwrite=True)

class PostPreselProcessor(PostProcessor):
    def __init__(self, ppcfg, luminosity=luminosity, groups=None, years=None) -> None:
        super().__init__(ppcfg, luminosity, groups, years)
    
    def hadd_results(self, *args, **kwargs):
        self._hadd_cutflows()
        self._hadd_csvouts()

    def _hadd_csvouts(self):
        """Hadd csv output files of datasets into one consolidated csv file for each year and group."""
        def process_csv(dsname, dtdir, outdir):
            """Process CSV files for a single dataset.

                Args:
                dsname: Dataset name
                dtdir: Input directory containing dataset files
                outdir: Output directory for processed files

                Returns:
                pd.DataFrame or None: Consolidated dataframe with metadata columns
            """
            try:
            # Load and concatenate CSV files
                dfs = DataLoader.load_csvs(
                    dirname=dtdir,
                    filepattern=f'{dsname}*output*csv',
                    func=lambda dfs: pd.concat(dfs, axis=0) if dfs else None
                )

                if dfs is not None:
                    return dfs
                else:
                    logging.info(f"No output files found for {dsname}")
                    return None

            except Exception as e:
                logging.error(f"Error processing CSV files for {dsname}: {e}")
                return None

    # Process datasets using DatasetIterator
        results = self.dataset_iter.process_datasets(process_csv)

        # Consolidate results by group
        for year, grouped in results.items():
            for group, nested in grouped.items():
                # Filter out None results and combine remaining dataframes
                valid_dfs = {k: v for k, v in nested.items() if v is not None}
                if valid_dfs:
                    # Add metadata columns
                    for dsname, df in valid_dfs.items():
                        df['dataset'] = dsname
                        df['year'] = year
                        df['group'] = group

                    # Combine all dataframes for this group
                    group_df = pd.concat(valid_dfs.values(), axis=0)

                    # Save consolidated output
                    outdir = pjoin(self.tempdir, year, group)
                    FileSysHelper.checkpath(outdir, createdir=True)
                    group_df.to_csv(pjoin(outdir, f'{group}_output.csv'))
                    logging.info(f"Saved consolidated output for {group} year {year}")
                else:
                    logging.warning(f"No valid dataframes found for group {group} year {year}")

    def check_results(self):
        pass
class PostSkimProcessor(PostProcessor):
    def __init__(self, ppcfg, luminosity, groups=None, years=None) -> None:
        super().__init__(ppcfg, luminosity, groups, years)
    
    def _init_metadata(self):
        """Load metadata for each year."""
        self.meta_dict = {}
        if self.cfg['IS_MC']:
            query_dir = pjoin(self.cfg['DATA_DIR'], 'availableMC')
        else:
            query_dir = pjoin(self.cfg['DATA_DIR'], 'availableData')
        logging.debug(f"Using metadata json file {query_dir}")
        years_to_process = []
        for year in self.years:
            if os.path.exists(pjoin(query_dir, f"{year}.json")):
                logging.debug(f"Loading metadata for {year}")
                with open(pjoin(query_dir, f"{year}.json"), 'r') as f:
                    self.meta_dict[year] = json.load(f)
                years_to_process.append(year)
            else:
                logging.warning(f"Metadata file not found for {year}. Removing from processing.")

        self.years = years_to_process
        logging.info(f"the following years are loaded: {self.meta_dict.keys()}")
    
    def check_and_clean(self):
        self.__check_roots()
        self.__clean_roots()
    
    def check_results(self):
        self.__check_roots()
    
    def clean_results(self):
        self.__clean_roots()
    
    def hadd_results(self):
        self.__hadd_roots()
        self._hadd_cutflows()
        if self.cfg['IS_MC']:
            logging.debug("Reporting total number of weighted events.")
            self.__get_total_nwgt_events()
    
    def __clean_roots(self):
        """Delete the corrupted files in the filelist."""
        filenames = FileSysHelper.glob_files(os.getcwd(), f"{self.cfg['DIRNAME']}_corrupted*.json")
        for filename in filenames:
            with open(filename, 'r') as f:
                all_corrupted = json.load(f)
            self.__delete_corrupted(all_corrupted)
            os.remove(filename)
            logging.info(f"Deleted corrupted files list {filename}")
    
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
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.cfg['DIRNAME']}_corrupted_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(corrupted, f)
            logging.error(f"Saved corrupted files to {filename}")
        
        return corrupted

    def __hadd_roots(self) -> str:
        """Hadd root files of datasets into appropriate size based on settings"""
        def process_ds(dsname, dtdir, outdir):
            root_files = FileSysHelper.glob_files(dtdir, f'{dsname}*.root', add_prefix=True, exclude='empty')
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

    def __delete_corrupted(self, results_dict):
        """Delete the corrupted files in the filelist."""
        for year in results_dict:
            for group in results_dict[year]:
                all_mismatched = results_dict[year][group]['mismatched_events']
                root_files = list(chain(*[d.get('root') for d in all_mismatched]))
                if root_files:
                    logging.debug(f"Deleting corrupted files: {root_files}")
                    FileSysHelper.remove_filelist(root_files)

                csv_files = [d.get('csv') for d in all_mismatched]
                if csv_files:
                    logging.debug(f"Deleting corrupted files: {csv_files}")
                    FileSysHelper.remove_filelist(csv_files)
        
    def __get_total_nwgt_events(self):
        """Calculate the sum of `Generator_weight` per each dataset and save to a json file with provided metadata.
        Updates existing metadata if present, otherwise creates new metadata file.

        The function will:
        1. Load existing metadata if present
        2. Update with new information from current processing
        3. Save the combined metadata back to file
        """
        new_meta_outpath = pjoin(self.cfg['DATA_DIR'], 'weightedMC')
        FileSysHelper.checkpath(new_meta_outpath, createdir=True)
        new_meta_dict = {}

        # Load existing metadata if present
        for year in self.years:
            meta_file = pjoin(new_meta_outpath, f"{year}.json")
            if os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    new_meta_dict[year] = json.load(f)
            else:
                new_meta_dict[year] = {}
                logging.debug(f"Metadata file not found for {year}. Creating new metadata table.")

        def get_nwgt_per_group(dsname, dtdir):
            resolved_df = pd.read_csv(FileSysHelper.glob_files(dtdir, f'*cf.csv')[0], index_col=0)
            if not resolved_df.filter(like=dsname).empty:
                nwgt = resolved_df.filter(like=dsname).filter(like='wgt').iloc[0,0]
                logging.debug(f"nwgt for {dsname} is {nwgt}")
                return nwgt
            else:
                logging.warning(f"{dsname} not found!")
                logging.debug(resolved_df)
                return None
        
        root_dtdir = self.tempdir if not self._will_trsf else self.transferP

        # Update metadata with new information
        for year, group, dsname, _ in self.dataset_iter.iterate_datasets(True):
            logging.debug(f"Processing {year}, {group}, {dsname}")
        # Initialize group dictionary if it doesn't exist
            if group not in new_meta_dict[year]:
                logging.debug(f'{group} not detected in currently loaded weighted stats!')
                new_meta_dict[year][group] = {}
            
            logging.debug(new_meta_dict[year][group])

            if dsname not in new_meta_dict[year][group]:
                logging.debug(f'{dsname} not detected in currently loaded weighted stats!')
                new_meta_dict[year][group][dsname] = self.meta_dict[year][group][dsname].copy()
            
            datadir = pjoin(root_dtdir, year, group)
            shortname = new_meta_dict[year][group][dsname]['shortname'] 
            nwgt = get_nwgt_per_group(shortname, datadir)
            if nwgt is not None:
                new_meta_dict[year][group][dsname]['nwgt'] = nwgt

        for year, items in new_meta_dict.items():
            logging.debug(f"Saving metadata for {year}")
            with open(pjoin(new_meta_outpath, f"{year}.json"), 'w') as f:
                json.dump(items, f, indent=4)
