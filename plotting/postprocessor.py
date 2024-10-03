import uproot, json, random, subprocess, gzip
import awkward as ak
import pandas as pd

from src.analysis.objutil import Object
from src.utils.filesysutil import FileSysHelper, pjoin, pbase
from src.utils.cutflowutil import combine_cf, calc_eff, load_csvs

class PostProcessor():
    """Class for loading and hadding data from skims/predefined selections produced directly by Processor.
    
    Attributes
    - `cfg`: the configuration file for post processing of datasets
    - `inputdir`: the directory where the input files are stored
    - `meta_dict`: the metadata dictionary for all datasets. {Groupname: {Datasetname: {metadata}}}"""
    def __init__(self, ppcfg, groups=None) -> None:
        """Parameters
        - `ppcfg`: the configuration file for post processing of datasets"""
        self.cfg = ppcfg
        self.lumi = ppcfg.LUMI * 1000
        self.inputdir = ppcfg.INPUTDIR
        self.tempdir = ppcfg.LOCALOUTPUT
        self.transferP = ppcfg.get("TRANSFERPATH", None)

        FileSysHelper.checkpath(self.inputdir, createdir=False, raiseError=True)
        FileSysHelper.checkpath(self.tempdir, createdir=True, raiseError=False)
        
        with open(ppcfg.METADATA, 'r') as f:
            self.meta_dict = json.load(f)

        self.groups = groups if not None else self.meta_dict.keys()

    def __call__(self, output_type=None, outputdir=None):
        """Hadd the root/csv files of the datasets and save to the output directory."""
        if output_type is None:
            output_type = self.cfg.get("OUTTYPE", 'root')
        if outputdir is None:
            outputdir = self.tempdir if self.transferP is None else self.transferP

        # self.hadd_cfs()
        if output_type == 'root': 
            # self.hadd_roots()
            self.meta_dict = PostProcessor.calc_wgt(outputdir, self.meta_dict, self.cfg.NEWMETA, self.groups)
        elif output_type == 'csv': PostProcessor.hadd_csvouts()
        else: raise TypeError("Invalid output type. Please choose either 'root' or 'csv'.")
    
    def check_roots(self):
        helper = FileSysHelper()
        for group in self.groups:
            possible_corrupted = []
            for root_file in helper.glob_files(pjoin(self.inputdir, group), '*.root'):
                try: 
                    with uproot.open(root_file) as f:
                        f.keys()
                except Exception as e:
                    print(f"Error reading file {root_file}: {e}")
                    possible_corrupted.append(root_file)
            if possible_corrupted:
                print(f"There are corrupted files for {group}!")
                with open(f'{group}_corrupted_files.txt', 'w') as f:
                    f.write('\n'.join(possible_corrupted))
            else:
                print(f"No corrupted files for {group} : > : > : > : >")
    
    @staticmethod
    def delete_corrupted(filelist_path):
        with open(filelist_path, 'r') as f:
            filelist = f.read().splitlines()
        FileSysHelper.remove_filelist(filelist)

    def __iterate_meta(self, callback):
        """Iterate over datasets and apply the callback function.
        
        Parameters
        - `callback`: the function to apply to each dataset. Expects output in the temp directory."""
        for group in self.groups:
            datasets = self.meta_dict[group]
            transferP = f"{self.transferP}/{group}" if self.transferP else None
            for _, dsitems in datasets.items():
                dsname = dsitems['shortname']
                dtdir = pjoin(self.inputdir, group)
                outdir = pjoin(self.tempdir, group)
                FileSysHelper.checkpath(outdir, createdir=True)
                if not FileSysHelper.checkpath(dtdir, createdir=False): continue
                callback(dsname, dtdir, outdir)
                if transferP is not None:
                    FileSysHelper.transfer_files(outdir, transferP, remove=True, overwrite=True)
    
    def hadd_roots(self) -> str:
        """Hadd root files of datasets into appropriate size based on settings"""
        def process_ds(dsname, dtdir, outdir):
            root_files = FileSysHelper.glob_files(dtdir, f'{dsname}*.root', add_prefix=True)
            batch_size = self.cfg.get("BATCHSIZE", 200)
            for i in range(0, len(root_files), batch_size):
                batch_files = root_files[i:i+batch_size]
                outname = pjoin(outdir, f"{dsname}_{i//batch_size+1}.root") 
                try:
                    call_hadd(outname, batch_files)
                except Exception as e:
                    print(f"Hadding encountered error {e}")
                    print(batch_files)
        
        self.__iterate_meta(process_ds)

    def hadd_csvouts(self) -> None:
        def process_csv(dsname, dtdir, outdir):
            concat = lambda dfs: pd.concat(dfs, axis=0)
            try:
                df = load_csvs(dtdir, f'{dsname}_output', func=concat)
                df.to_csv(pjoin(outdir, f"{dsname}_out.csv"))
            except Exception as e:
                print(f"Error loading csv files for {dsname}: {e}")
        
        self.__iterate_meta(process_csv)
        
    def hadd_cfs(self):
        """Hadd cutflow table output from processor"""
        def process_cf(dsname, dtdir, outdir):
            print(f"Dealing with {dsname} now ...............................")
            try:
                df = combine_cf(inputdir=dtdir, dsname=dsname, output=False)
                df.to_csv(pjoin(outdir, f"{dsname}_cf.csv"))
            except Exception as e:
                print(f"Error combining cutflow tables for {dsname}: {e}")
        
        self.__iterate_meta(process_cf)

    # not usable for now
    # @staticmethod
    # def check_cf(groupnames, base_dir) -> None:
    #     """Check the cutflow numbers against the number of events in the root files."""
    #     for group in groupnames:
    #         query_dir = pjoin(base_dir, group)
    #         cf = PostProcessor.load_cf(group, base_dir)[0]
    #         if check_last_no(cf, f"{process}_raw", glob_files(condorpath, f'{process}*.root')):
    #             print(f"Cutflow check for {process} passed!")
    #         else:
    #             print(f"Discrepancies between cutflow numbers and output number exist for {process}. Please double check selections.")
                
    def merge_cf(self, inputdir=None, outputdir=None) -> pd.DataFrame:
        """Merge all cutflow tables for all processes into one. Save to LOCALOUTPUT.
        Output formatted cutflow table as well.
        
        Parameters
        - `signals`: list of signal process names
        
        Return 
        - dataframe of weighted cutflows for every dataset merged"""
        if inputdir is None: inputdir = self.inputdir
        else: FileSysHelper.checkpath(inputdir, createdir=False, raiseError=True)
        
        if outputdir is None: outputdir = self.tempdir
        else: FileSysHelper.checkpath(outputdir, createdir=True)

        resolved_list = []
        for group in self.groups:
            resolved, _ = PostProcessor.load_cf(group, self.meta_dict, inputdir) 
            resolved_list.append(resolved)
        resolved_all = pd.concat(resolved_list, axis=1)
        resolved_all.to_csv(pjoin(outputdir, "allDatasetCutflow.csv"))
        wgt_resolved = resolved_all.filter(like='wgt', axis=1)
        wgt_resolved.columns = wgt_resolved.columns.str.replace('_wgt$', '', regex=True)
        wgt_resolved.to_csv(pjoin(outputdir, "ResolvedWgtOnly.csv"))
        wgtpEff = calc_eff(wgt_resolved, None, 'incremental', True)
        wgtpEff.filter(like='eff', axis=1).to_csv(pjoin(outputdir, "ResolvedEffOnly.csv"))

        return wgt_resolved
    
    @staticmethod
    def present_yield(wgt_resolved, signals, outputdir, regroup_dict=None) -> pd.DataFrame:
        """Present the yield dataframe with grouped datasets. Regroup if necessary.
        
        Parameters
        - `signals`: list of signal group names
        - `regroup_dict`: dictionary of regrouping keywords. Passed into `PostProcessor.categorize`.
        """
        if regroup_dict is not None:
            wgt_resolved = PostProcessor.categorize(wgt_resolved, regroup_dict)
        
        yield_df = PostProcessor.process_yield(wgt_resolved, signals)
        yield_df.to_csv(pjoin(outputdir, 'scaledyield.csv'))
        
        return yield_df
    
    @staticmethod
    def process_yield(yield_df, signals) -> pd.DataFrame:
        """group the yield dataframe to include signal and background efficiencies.

        Parameters
        - `yield_df`: dataframe of yields
        - `signals`: list of signal group names
        
        Return
        - processed yield dataframe"""
        sig_list = [signal for signal in signals if signal in yield_df.columns]
        bkg_list = yield_df.columns.difference(sig_list)

        yield_df['Tot Bkg'] = yield_df[bkg_list].sum(axis=1)
        yield_df['Bkg Eff'] = calc_eff(yield_df, 'Tot Bkg', inplace=False)

        for signal in sig_list:
            yield_df[f'{signal} Eff'] = calc_eff(yield_df, signal, inplace=False)

        new_order = list(bkg_list) + ['Tot Bkg', 'Bkg Eff']
        for signal in sig_list:
            new_order.extend([signal, f'{signal} Eff'])
            
        yield_df = yield_df[new_order]
        return yield_df
    
    @staticmethod
    def categorize(df, group_kwd:'dict') -> pd.DataFrame:
        """Recalculate/categorize a table by the group keyword.
        
        Parameters
        - `group_kwd`: {name of new column: [keywords to search for in the column names]}"""
        for newcol, kwdlist in group_kwd.items():
            cols = [col for col in df.columns if any(kwd in col for kwd in kwdlist)]
            if cols:
                df[newcol] = df[cols].sum(axis=1)
                df.drop(columns=cols, inplace=True)
        return df
    
    @staticmethod
    def calc_wgt(datasrcpath, meta_dict, dict_outpath, groups) -> dict:
        """Calculate the weight per event for each dataset and save to a json file with provided metadata.
        
        Parameters
        - `datasrcpath`: path to the output directory (base level)
        - `meta_dict`: metadata dictionary for all datasets. {group: {dataset: {metadata}}}"""
        for group in groups:
            print(f"Globbing from {pjoin(datasrcpath, group)}")
            resolved_df = pd.read_csv(FileSysHelper.glob_files(pjoin(datasrcpath, group), f'{group}*cf.csv')[0], index_col=0) 
            for ds, dsitems in meta_dict[group].items():
                nwgt = resolved_df.filter(like=dsitems['shortname']).filter(like='wgt').iloc[0,0]
                meta_dict[group][ds]['nwgt'] = nwgt
                meta_dict[group][ds]['per_evt_wgt'] = meta_dict[group][ds]['xsection'] / nwgt
        
        with open(pjoin(dict_outpath), 'w') as f:
            json.dump(meta_dict, f)
        
        return meta_dict

    @staticmethod
    def load_cf(group, meta_dict, datasrcpath) -> tuple[pd.DataFrame]:
        """Load cutflow tables for one group containing datasets to be grouped tgt and scale it by xsection 

        Parameters
        -`group`: the name of the cutflow that will be grepped from datasrcpath
        -`datasrcpath`: path to the output directory (base level)
        
        Returns
        - tuple of resolved (per channel) cutflow dataframe and combined cutflow (per group) dataframe"""
        resolved_df = pd.read_csv(FileSysHelper.glob_files(pjoin(datasrcpath, group), f'{group}*cf.csv')[0], index_col=0)
        for _, dsitems in meta_dict[group].items():
            dsname = dsitems['shortname']
            per_evt_wgt = dsitems['per_evt_wgt']
            sel_cols = resolved_df.filter(like=dsname).filter(like='wgt')
            resolved_df[sel_cols.columns] = sel_cols * per_evt_wgt
            combined_cf = PostProcessor.sum_kwd(resolved_df, 'wgt', f"{group}_wgt")
        return resolved_df, combined_cf

    @staticmethod
    def sum_kwd(cfdf, keyword, name) -> pd.Series:
        """Add a column to the cutflow table by summing up all columns with the keyword.

        Parameters
        - `cfdf`: cutflow dataframe
        - `keyword`: keyword to search for in the column names
        - `name`: name of the new column

        Return
        - Series of the summed column"""
        same_cols = cfdf.filter(like=keyword)
        sumcol = same_cols.sum(axis=1)
        cfdf = cfdf.drop(columns=same_cols)
        cfdf[name] = sumcol
        return sumcol

    @staticmethod
    def write_obj(writable, filelist, objnames, extra=[]) -> None:
        """Writes the selected, concated objects to root files.
        Parameters:
        - `writable`: the uproot.writable directory
        - `filelist`: list of root files to extract info from
        - `objnames`: list of objects to load. Required to be entered in the selection config file.
        - `extra`: list of extra branches to save"""

        all_names = objnames + extra
        all_data = {name: [] for name in objnames}
        all_data['extra'] = {name: [] for name in extra}
        for file in filelist:
            evts = load_fields(file)
            print(f"events loaded for file {file}")
            for name in all_names:
                if name in objnames:
                    obj = Object(evts, name)
                    zipped = obj.getzipped()
                    all_data[name].append(zipped)
                else:
                    all_data['extra'][name].append(evts[name])
        for name, arrlist in all_data.items():
            if name != 'extra':
                writable[name] = ak.concatenate(arrlist)
            else:
                writable['extra'] = {branchname: ak.concatenate(arrlist[branchname]) for branchname in arrlist.keys()}
    
    @staticmethod
    def process_file(path, group, resolution):
        """Read and group a file based on resolution."""
        df = pd.read_csv(path, index_col=0)
        if resolution == 0:
            df = df.sum(axis=1).to_frame(name=group)
        return df

def check_last_no(df, col_name, rootfiles):
    """Check if the last number in the cutflow table matches the number of events in the root files.
    
    Parameters
    - `df`: cutflow dataframe
    - `col_name`: name of the column to check
    - `rootfiles`: list of root files
    """
    if isinstance(rootfiles, str):
        rootfiles = [rootfiles]
    
    raw = 0
    for file in rootfiles:
        with uproot.open(file) as f:
            thisraw = f.get("Events").num_entries
        raw += thisraw
    
    print(f'Got {raw} events in root files!')
    print(f'Got {df[col_name].iloc[-1]} events in cutflow table!')
    
    return df[col_name].iloc[-1] == raw

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

def load_fields(file, branch_names=None, tree_name='Events', lib='ak') -> tuple[ak.Array, list]:
    """Load specific fields if any. Otherwise load all. If the file is a list, concatenate the data from all files.
    
    Parameters:
    - file: path to the root file or list of paths
    - branch_names: list of branch names to load
    - tree_name: name of the tree in the root file
    - lib: library to use for loading the data

    Returns:
    - awkward array of the loaded data (, list of empty files)
    """
    def load_one(fi):
        with uproot.open(fi) as file:
            if file.keys() == []:
                return False
            else:
                tree = file[tree_name] 
        return tree.arrays(branch_names, library=lib)

    returned = None
    if isinstance(file, str):
        file = [file]
    dfs = []
    emptylist = []
    for root_file in file:
        result = load_one(root_file)
        if result: dfs.append(load_one(file))
        else: emptylist.append(root_file)
    combined_evts = ak.concatenate(dfs)
    return combined_evts, emptylist

def write_root(evts: 'ak.Array | pd.DataFrame', destination, outputtree="Events", title="Events", compression=None):
    """Write arrays to root file. Highly inefficient methods in terms of data storage.

    Parameters
    - `destination`: path to the output root file
    - `outputtree`: name of the tree to write to
    - `title`: title of the tree
    - `compression`: compression algorithm to use"""
    branch_types = {name: evts[name].type for name in evts.fields}
    with uproot.recreate(destination, compression=compression) as file:
        file.mktree(name=outputtree, branch_types=branch_types, title=title)
        file[outputtree].extend({name: evts[name] for name in evts.fields}) 

def call_hadd(output_file, input_files):
    """Merge ROOT files using hadd.
    Parameters
    - `output_file`: path to the output file
    - `input_files`: list of paths to the input files"""
    command = ['hadd', '-f0 -O', output_file] + input_files
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Merged files into {output_file}")
    else:
        print(f"Error merging files: {result.stderr}")    
