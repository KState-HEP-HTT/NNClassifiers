import sys
import pandas as pd
import numpy as np
from glob import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler

## Variables used for selection. These shouldn't be normalized
selection_vars = [
    'cat_vbf', 'el_charge', 't1_charge', 'mu_charge', 'nbjets'
]

## Variables that could be used as NN input. These should be normalized
scaled_vars = [
    'evtwt', 'mt', 'njets',
    'mjj', 'dEtajj', 'm_sv', 'pt_sv', 'el_pt', 't1_pt', 'mu_pt', 't2_pt', 'hjj_pT', 'higgs_pT',
    'Dbkg_VBF', 'Dbkg_ggH', 'Q2V1', 'Q2V2', 'Phi', 'Phi1', 'costheta1', 'costheta2', 'costhetastar'
]

def loadFile(ifile):
    from root_pandas import read_root

    if 'MUTAU' in ifile:
        channel = 'mt'
    elif 'ETAU' in ifile:
        channel = 'et'
    elif 'TAUTAU' in ifile:
        channel = 'tt'
    else:
        raise Exception('Input files must have MUTAU, ETAU, or TAUTAU in the provided path. You gave {}, ya goober.'.format(ifile))

    filename = ifile.split('/')[-1]
    print 'Loading input file...', filename

    input_df = read_root(ifile, columns=scaled_vars+selection_vars) ## read from TTrees into DataFrame
    slim_df = input_df[(input_df['njets'] > 1) & (input_df['mjj'] > 300) & (input_df['mt'] < 50)] ## preselection
    selection_df = slim_df[selection_vars] ## get variables needed for selection (so they aren't normalized)
    weights = slim_df['evtwt'] ## get just the weights (they are scaled differently)
    slim_df = slim_df.drop(selection_vars+['evtwt'], axis=1)

    ## add the event label
    if 'VBF' in ifile or 'ggH' in ifile:
        isSignal = np.ones(len(slim_df))
    else:
        isSignal = np.zeros(len(slim_df))

    ## save the name of the process
    somenames = np.full(len(slim_df), filename.split('.root')[0])

    ## scale event weights between 0 - 1
    weights = MinMaxScaler().fit_transform(weights.values.reshape(-1,1))

    ## get lepton channel
    lepton = np.full(len(slim_df), channel)

    return slim_df, selection_df, somenames, lepton, isSignal, weights


def main(args):
    input_files = [ifile for ifile in glob('{}/*.root'.format(args.el_input_dir))]
    input_files += [ifile for ifile in glob('{}/*.root'.format(args.mu_input_dir)) if args.mu_input_dir != None]
    input_files += [ifile for ifile in glob('{}/*.root'.format(args.tau_input_dir)) if args.tau_input_dir != None]

    ## define collections that will all be merged in the end
    big_boi, selection_boi = pd.DataFrame(), pd.DataFrame()
    names, leptons, isSignal, weight_boi = np.array([]), np.array([]), np.array([]), np.array([])

    for ifile in input_files:
        input_data, selection_data, new_name, lepton, sig, weight = loadFile(ifile)
        big_boi = pd.concat([big_boi, input_data])  ## add data to the full set
        selection_boi = pd.concat([selection_boi, selection_data]) ## add selection variables to full set
        names = np.append(names, new_name) ## insert the name of the current sample
        isSignal = np.append(isSignal, sig) ## labels for signal/background
        weight_boi = np.append(weight_boi, weight) ## weights scaled from 0 - 1
        leptons = np.append(leptons, lepton) ## lepton channel

    ## normalize the potential training variables
    skinny_boi = pd.DataFrame(StandardScaler().fit_transform(big_boi.values), columns=big_boi.columns.values)

    ## add selection variables
    for column in selection_boi.columns:
        skinny_boi[column] = selection_boi[column].values

    ## add other useful data
    skinny_boi['sample_names'] = pd.Series(names)
    skinny_boi['lepton'] = pd.Series(leptons)
    skinny_boi['isSignal'] = pd.Series(isSignal)
    skinny_boi['evtwt'] = pd.Series(weight_boi)

    ## save the dataframe for later
    store = pd.HDFStore('datasets/{}.h5'.format(args.output))
    store['df'] = skinny_boi

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--el-input', '-e', action='store', dest='el_input_dir', default=None, help='path to etau input files')
    parser.add_argument('--mu-input', '-m', action='store', dest='mu_input_dir', default=None, help='path to mutau input files')
    parser.add_argument('--tau-input', '-t', action='store', dest='tau_input_dir', default=None, help='path to tautau input files')
    parser.add_argument('--output', '-o', action='store', dest='output', default='store.h5', help='name of output file')

    main(parser.parse_args())
