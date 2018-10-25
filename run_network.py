#!/usr/bin/env python

from argparse import ArgumentParser
parser = ArgumentParser(
  description="Run neural network to separate Z->TT + 2 jets from VBF"
  )
parser.add_argument('--verbose', action='store_true',
                    dest='verbose', default=False,
                    help='run in verbose mode'
                    )
parser.add_argument('--input', '-i', action='store',
                    dest='input', default=None, type=str,
                    help='File with events to input to NN'
                    )
parser.add_argument('--load_json', '-l', action='store',
                    dest='load_json', default='model_store.json',
                    help='name of json file to load model parameters'
                    )
parser.add_argument('--treename', '-t', action='store',
                    dest='treename', default='etau_tree',
                    help='name of the tree to read'
                    )
args = parser.parse_args()

import os
import sys
import json
import pandas
from os import environ
from root_pandas import read_root
environ['KERAS_BACKEND'] = 'tensorflow'  ## on Wisc machine, must be before Keras import
from keras.models import Model
from keras.layers import Input, Dense

## this function needs serious refactoring
def putInTree(fname, discs, all_discs):
  """Function to write a new file copying old TTree and adding NN discriminant"""
  from ROOT import TFile
  from array import array
  fin = TFile(fname, 'read')
  itree = fin.Get(args.treename) 
  oname = fname.split('/')[-1].split('.root')[0]
  fout = TFile('output_files/'+oname+'.root', 'recreate')  ## make new file for output
  fout.cd()
  nevents = fin.Get("nevents").Clone()
  nevents.Write()
  ntree = itree.CloneTree(-1, 'fast')  ## copy all branches from old tree
  adiscs = array('f', [0.])
  disc_branch = ntree.Branch('NN_disc', adiscs, 'NN_disc/F')  ## make a new branch to store the disc

  i, j = 0, 0
  for event in itree:
    if event.Q2V1 > 0 and event.cat_vbf > 0 and event.el_charge + event.t1_charge == 0 and event.nbjets == 0:
      adiscs[0] = discs[j][0]
      j += 1
    else:
      adiscs[0] = all_discs[i][0]
    i += 1
    fout.cd()
    disc_branch.Fill()
  
  fin.Close()
  fout.cd()
  ntree.Write()

def build_network(model_name):
  print 'Building the network...'
  inputs = Input(shape = (input_length,), name = 'input')
  hidden = Dense(nhid, name = 'hidden', kernel_initializer = 'normal', activation = 'sigmoid')(inputs)
#  hidden1 = Dense(4, name = 'hidden1', kernel_initializer = 'normal', activation = 'sigmoid')(inputs)
#  hidden2 = Dense(4, name = 'hidden2', kernel_initializer = 'normal', activation = 'sigmoid')(hidden1)
  outputs = Dense(1, name = 'output', kernel_initializer = 'normal', activation = 'sigmoid')(hidden)
  model = Model(inputs = inputs, outputs = outputs)
  model.load_weights(model_name)
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model

def create_dataframe(variables):
 ## begin section to run a trained NN on all events in a file
  print 'Loading data...'
  
  ## read necessary branches from input file
  df = read_root(args.input, columns=(variables + ['cat_vbf', 'el_charge', 't1_charge', 'nbjets', 'mjj']))
  all_data = read_root(args.input, columns=(variables))
  df = df[(df['Q2V1'] > 0) & (df['cat_vbf'] > 0) & (df['el_charge'] + df['t1_charge'] == 0) & (df['nbjets'] == 0)]
  df = df.drop(['cat_vbf', 'el_charge', 't1_charge', 'nbjets', 'mjj'], axis=1)
  return df, all_data

def normalize(df):
  """Take a pandas DataFrame and normalize the variables"""
  from sklearn.preprocessing import StandardScaler
  return StandardScaler().fit_transform(df.values)

if __name__ == "__main__":

  with open(args.load_json, 'r') as fname:
    params = json.load(fname)

  model_name = params['model_name']
  variables  = params['variables']
  nhid       = params['nhidden']
  input_length = len(variables)
  model_name = 'models/' + model_name + '.hdf5'
  if not os.path.exists(model_name):
    print "Can't find trained model: {}".format(model_name)
    print "Please run train_network.py -m {}  to train a model named {}".format(model_name, model_name)
    sys.exit(0)

  ## build the network
  model = build_network(model_name)

  ## load data and do event selection
  df, all_df = create_dataframe(variables)

  ## normalize the data
  data = normalize(df)
  all_data = normalize(all_df)

  ## run the NN and make prediction for all events
  print 'Making predictions...'
  predict = model.predict(data, verbose=args.verbose)
  predict_all = model.predict(all_data, verbose=False)
  
  ## put NN discriminant into TTree
  print 'Filling the tree...'
  putInTree(args.input, predict, predict_all)      
  print 'Done.'
