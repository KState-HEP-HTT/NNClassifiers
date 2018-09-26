#!/usr/bin/env python

from argparse import ArgumentParser
parser = ArgumentParser(
  description = "Train two-layer neural network to separate Drell-Yan + jets from VBF Higgs to tau-tau"
  )
parser.add_argument('--verbose', action='store_true',
                    dest='verbose', default=False,
                    help='run in verbose mode'
                    )
parser.add_argument('--nhid', '-n', action='store',
                    dest='nhid', default=5, type=int,
                    help='number of hidden nodes in network'
                    )
parser.add_argument('--vars', '-v', nargs='+', action='store',
                    dest='vars', default=['Q2V1', 'Q2V2'],
                    help='variables to input to network'
                    )
parser.add_argument('--njet', '-N', action='store_true',
                    dest='njet', default=False,
                    help='run on DY + n-jets'
                    )
parser.add_argument('--model_name', '-m', action='store',
                    dest='model_name', default=None,
                    help='name of a trained model'
                    )
parser.add_argument('--dont_save_json', '-d', action='store_true',
                    dest='dont_save_json', default=False,
                    help="don't store NN settings to json"
                    )

args = parser.parse_args()
n_user_inputs = 0

import h5py
import pandas
import numpy as np
from os import environ
from pprint import pprint
from root_pandas import read_root
environ['KERAS_BACKEND'] = 'tensorflow'  ## on Wisc machine, must be before Keras import
from keras.models import Model
from keras.layers import Input, Activation, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

def create_json(model_name):
  import json

  if args.model_name != None:
    fname = args.model_name + '.json'
  else:
    fname = 'model_store.json'

  with open('model_params/'+fname, 'w') as fout:
    json.dump(
      {
        'model_name': model_name,
        'variables': args.vars,
        'nhidden': args.nhid,
        'njet': args.njet,
        'n_user_inputs': int(n_user_inputs)
      }, fout
    )

def build_nn(nhid):
  """ Build and return the model with callback functions """

  print 'Building the network...'
  inputs = Input(shape = (input_length,), name = 'input')
  hidden = Dense(nhid, name = 'hidden', kernel_initializer = 'normal', activation = 'sigmoid')(inputs)
  outputs = Dense(1, name = 'output', kernel_initializer = 'normal', activation = 'sigmoid')(hidden)
  model = Model(inputs = inputs, outputs = outputs)

  # early stopping callback
  early_stopping = EarlyStopping(monitor='val_loss', patience=10)

  if args.model_name != None:
    model_name = args.model_name + '.hdf5'
  else:
    if args.njet:
      model_name = 'NN_njet_model.hdf5'
    else:
      model_name = 'NN_2jet_model.hdf5'

  model_name = 'models/' + model_name

  # model checkpoint callback
  model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss',
                      verbose=0, save_best_only=True,
                      save_weights_only=False, mode='auto',
                      period=1)
  callbacks = [early_stopping, model_checkpoint]

  ## compile the model and return it with callbacks
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  print 'Build complete.'
  return model, callbacks

def massage_data(vars, fname, sample_type):
  """ read input h5 file, slice out unwanted data, return DataFrame with variables and one-hot """
  from ROOT import TLorentzVector

  print 'Slicing and dicing...', fname.split('.root')[0].split('input_files/')[-1]
  other_vars = ['evtwt', 'passSelection', 'Dbkg_VBF']
  slicer = vars + other_vars  ## add variables for event selection
  
  ## Switch to using root_pandas
  df = read_root(fname, columns=slicer) 
  df_roc = pandas.DataFrame()
  df_roc['Dbkg_VBF'] = df[(df['passSelection'] > 0) & (df['Dbkg_VBF'] > 0)]['Dbkg_VBF']
  
  print 'Input data is now in the DataFrame'

  ## additional variables for selection that must be constructed can be added here

  ## define additional event selection

  print 'Begin the slicing'

  qual_cut = (df['Q2V1'] > 0) & (df['passSelection'] > 0) & (df['evtwt'] > 0)
  df = df[qual_cut]

  if 'bkg' in sample_type:

    ## format background DataFrame for NN input
    df['isSignal'] = np.zeros(len(df))  ## put label in DataFrame (bkg=0)

    ## format bkg DataFrame for MELA ROC curve
    df_roc['isSignal'] = np.zeros(len(df_roc))

  else:

    ## format signal DataFrame for NN input
    df['isSignal'] = np.ones(len(df))  ## put label in DataFrame (sig=1)

    ## format bkg DataFrame for MELA ROC curve
    df_roc['isSignal'] = np.ones(len(df_roc))
    
  ## additional input variables that must be constructed can be added here
  #df = AddInput(db, 'dEtajj', abs(df['jeta_1'] - df['jeta_2']))

  weight = df['evtwt']
  df.insert(loc=df.shape[1], column='weight', value=weight)
  
  ## drop event selection branches from NN input
  df = df.drop(other_vars, axis=1)
  return df, df_roc


def AddInput(dataf, name, val):
   dataf.insert(loc=dataf.shape[1], column=name, value=val)
   global n_user_inputs
   n_user_inputs += .5
   return dataf

def build_plots(history, label_test, other=None):
  """ do whatever plotting is needed """
  import  matplotlib.pyplot  as plt
  from sklearn.metrics import roc_curve, auc

  plt.figure(figsize=(15,10))

  # Plot ROC
  label_predict = model.predict(data_test)
  fpr, tpr, thresholds = roc_curve(label_test[:,0], label_predict[:,0])
  roc_auc = auc(fpr, tpr)
  plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
  plt.plot(tpr, fpr, lw=2, color='cyan', label='NN auc = %.3f' % (roc_auc))
  if other != None:
    fpr2, tpr2, thresholds2, roc_auc2 = other
    plt.plot(tpr2, fpr2, lw=2, color='red', label='MELA auc = %.3f' % (roc_auc2))
  plt.xlim([0, 1.0])
  plt.ylim([0, 1.0])
  plt.xlabel('true positive rate')
  plt.ylabel('false positive rate')
  plt.title('receiver operating curve')
  plt.legend(loc="upper left")
  plt.savefig('plots/'+args.model_name+'.pdf')
  #plt.show()
  # if args.njet:
  #   ext = 'njet'
  #   plt.savefig('plots/layer1_node{}_njet_NN.pdf'.format(args.nhid))
  # else:
  #   plt.savefig('plots/layer1_node{}_2jet_NN.pdf'.format(args.nhid))

  # plot loss vs epoch
  ax = plt.subplot(2, 1, 1)
  ax.plot(history.history['loss'], label='loss')
  ax.plot(history.history['val_loss'], label='val_loss')
  ax.legend(loc="upper right")
  ax.set_xlabel('epoch')
  ax.set_ylabel('loss')

  # plot accuracy vs epoch
  ax = plt.subplot(2, 1, 2)
  ax.plot(history.history['acc'], label='acc')
  ax.plot(history.history['val_acc'], label='val_acc')
  ax.legend(loc="upper left")
  ax.set_xlabel('epoch')
  ax.set_ylabel('acc')
  plt.show()

def final_formatting(data, labels):
  """ split data into testing and validation then scale and return collections """
  from sklearn.preprocessing import StandardScaler
  from sklearn.model_selection import train_test_split

  ## split data into labels and also split into train/test
  data_train_val, data_test, label_train_val, label_test = train_test_split(data, labels, test_size=0.2, random_state=7)

  ## normalize all input variables to improve performance
  scaler = StandardScaler().fit(data_train_val)
  data_train_val = scaler.transform(data_train_val)
  data_test = scaler.transform(data_test)

  ## return the same that train_test_split does, but normalized
  return data_train_val, data_test, label_train_val, label_test

def MELA_ROC(sig, bkg):
  """ read h5 file and return info for making a ROC curve from MELA disc. """
  from sklearn.metrics import roc_curve, auc
  all_data = pandas.concat([sig, bkg])
  dataset = all_data.values
  data = dataset[:,0:1]  ## read Dbkg_VBF for all events
  labels = dataset[:,-1]  ## read labels for all events
  fpr, tpr, thresholds = roc_curve(labels, data[:,0])
  roc_auc = auc(fpr, tpr)  ## calculate Area Under Curve
  return fpr, tpr, thresholds, roc_auc

if __name__ == "__main__":
  ## format the data
  sig, mela_sig = massage_data(args.vars, "input_files/VBF125.root", "sig")
  input_length = sig.shape[1] - 2  ## get shape and remove weight & isSignal
  bkg, mela_bkg = massage_data(args.vars, "input_files/embed.root", "bkg")
  print 'Training Statistics ----'
  print 'No. Signal', sig.shape[0]
  print 'No. Backg.', bkg.shape[0]
  all_data = pandas.concat([sig, bkg])
  dataset = all_data.values
  data = dataset[:,0:input_length]  ## get numpy array with all input variables
  labels = dataset[:,input_length:]  ## get numpy array with weights and labels
  data_train_val, data_test, label_train_val, label_test = final_formatting(data, labels) ## split/normalize/randomize data
  label_train_val, weights = label_train_val[:,0], label_train_val[:,1]  ## split into labels and weights

  ## build the NN
  model, callbacks = build_nn(args.nhid)
  if args.verbose:
    model.summary()

  ## train the NN
  history = model.fit(data_train_val,
                    label_train_val,
                    epochs=5000,
                    batch_size=1024,
                    verbose=args.verbose, # switch to 1 for more verbosity
                    callbacks=callbacks,
                    validation_split=0.25,
                    sample_weight=weights
  )

  if args.verbose:
    ## produce a ROC curve and other plots
    build_plots(history, label_test, MELA_ROC(mela_sig, mela_bkg))

  if args.dont_save_json:
    pass
  else:
    if args.model_name != None:
      model_name = args.model_name
    elif args.njet:
      model_name = 'NN_njet_model'
    else:
      model_name = 'NN_2jet_model'

    create_json(model_name)
