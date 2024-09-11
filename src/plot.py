from plotting.postprocess import PostProcessor
from plotting.visutil import CSVPlotter
from config.plotsetting import object_dict
import argparse


def postprocess():
    pp = PostProcessor()
    pp()

def mergecf():
    cf_df = PostProcessor.merge_cf()
    PostProcessor.present_yield(cf_df, ['ZH', 'HH', 'ZH'])

# def checkouts():
#     PostProcessor.check_roots()

def plotouts():
    cp = CSVPlotter()
    df = cp.postprocess(per_evt_wgt='Generator_weight_values')
    cp.plot_hist(df, object_dict)

def programchoice() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Choose the post-processing options')
    parser.add_argument('--postprocess', action='store_true', help='Execute cutflow table merging')
    parser.add_argument('--mergecf', action='store_true', help='Execute hadding procedure for the specified, processed datasets')
    # parser.add_argument('--checkouts', action='store_true', help='Check the output root')
    parser.add_argument('--getobj', action='store_true', help='Get delimited object files')
    parser.add_argument('--plotouts', action='store_true', help='Plot stored options')

    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
        parser.exit()

    return args

if __name__ == '__main__':
    args = programchoice()
    if args.postprocess: postprocess()
    if args.mergecf: mergecf()
    # if args.checkouts: checkouts()
    if args.plotouts: plotouts()

