#!/usr/bin/env python
# adapted from: https://github.com/bu-cms/bucoffea/blob/83daf25146d883df5131d0b50a51c0a6512d7c5f/bucoffea/helpers/dasgowrapper.py

import json, shutil, argparse
from coffea.dataset_tools.dataset_query import DataDiscoveryCLI

class QueryRunner:
    """Class to run the query on the dataset and preprocess the data."""
    def __init__(self) -> None:
        self.ddc = DataDiscoveryCLI()
        self.ddc.do_regex_sites(r"T[123]_(US)_\w+")

    def __call__(self, dataset, infile='availableQuery.json') -> None:
        with open(infile, 'r') as file:
            mcstrings = json.load(file)
        
        self.ddc.load_dataset_definition(dataset_definition=mcstrings[dataset], query_results_strategy='all', replicas_strategy='manual')

        self.ddc.do_preprocess(output_file=dataset,
            step_size=10000,
            align_to_clusters=False,
            recalculate_steps=False,
            files_per_batch=1,
            file_exceptions=(OSError, IndexError),
            save_form=False,
            allow_empty_datasets=True,
            scheduler_url=None)
        
        shutil.move(f"{dataset}_available.json.gz", f"preprocessed/{dataset}.json.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run preprocessor on fileset')    
    parser.add_argument('-d', '--dataset', type=str, required=True, help='dataset to run program on')
    parser.add_argument('-s', '--skip', action='store_true', required=False, help='whether to skip preprocess.')

    args = parser.parse_args()
    qr = QueryRunner()
    if args.skip:
        qr.dump_query(args.dataset)
    else:
        qr(args.dataset)