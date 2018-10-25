import numpy as np
import pandas as pd
from glob import glob
from os import environ
from root_pandas import read_root
# on Wisc machine, must be before Keras import
environ['KERAS_BACKEND'] = 'tensorflow'
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from visualize import *

def create_json(model_name, nhid, vars):
    """  
    Save the structure of the model to a JSON file. This file is 
    read later by run_network.py to create the correct network when 
    doing the actual classifying 
    """
    import json

    if model_name != None:
        fname = model_name + '.json'
    else:
        fname = 'model_store.json'

    with open('model_params/'+fname, 'w') as fout:
        json.dump(
            {
                'model_name': model_name,
                'variables': vars,
                'nhidden': nhid[0],
                'n_user_inputs': len(vars)
            }, fout
        )

class Classifier:
    """
    Classifier holds all the data/functions needed from creating and training
    a network to do binary classification with an arbitrary number of hidden
    layers in a fully-connected network
    """
    def __init__(self, name, ninp, nhid):

        # initialize data holders
        self.name = name
        self.ninp = ninp
        self.sig = pd.DataFrame()
        self.bkg = pd.DataFrame()
        self.data = NotImplemented
        self.label = NotImplemented
        self.weight = NotImplemented
        self.callbacks = NotImplemented

        # first, build the model
        self.model = Sequential()

        # add the input layer
        self.model.add(
            Dense(nhid[0], input_shape=(nhid[0],), name='input', activation='sigmoid')
        )

        # add hidden layers
        for i in range(len(nhid)-1):
            self.model.add(
                Dense(nhid[i], activation='sigmoid', kernel_initializer='normal')
            )

        # last, add output layer
        self.model.add(
            Dense(1, activation='sigmoid', kernel_initializer='normal')
        )

        # print information while training
        self.model.summary()

        # then, compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy',
                           metrics=['accuracy'])

        # now add some callbacks
        self.callbacks = [
            EarlyStopping(monitor='val_loss', patience=50),
            ModelCheckpoint('models/'+name+'.hdf5', monitor='val_loss',
                            verbose=0, save_best_only=True,
                            save_weights_only=False, mode='auto',
                            period=1
                            )
        ]

    def loadData(self, vars, sig_name, bkg_name):
        # quick function for loading
        def load(name):
            ## variables to load for selection
            other_vars = [
                'evtwt', 'cat_inclusive', 'cat_0jet', 'cat_boosted', 'cat_vbf',
                'Dbkg_VBF', 'Dbkg_ggH', 'njets', 'higgs_pT', 't1_charge', 'el_charge', 'nbjets', 'mt'
            ]
            slicer = vars + other_vars  # add variables for event selection
            df = read_root(name, columns=slicer) ## only read specified branches from TTree

            # apply selection and make sure variables are reasonable
            qual_cut = (df['Q2V1'] > 0) & \
                       (df['cat_vbf'] > 0) & (df['nbjets'] == 0) & (df['mt'] < 50) & \
                       (df['el_charge'] + df['t1_charge'] == 0)
            df = df[qual_cut]

            # make sure the weight is in the correct column and is normalized 
            weight = df['evtwt'].values
            from sklearn.preprocessing import MinMaxScaler
            points = weight.shape[0]
            weight = weight.reshape(-1, 1)
            weight = MinMaxScaler(feature_range=(0, 1)).fit_transform(weight)
            weight = weight.reshape(points, -1)
            df.insert(loc=df.shape[1], column='weight', value=weight)

            # remove un-needed columns
            df = df.drop(other_vars, axis=1)

            return df

        self.sig = load(sig_name)
        self.bkg = load(bkg_name)

        ## add labels to the datasets
        self.sig['isSignal'] = np.ones(len(self.sig))
        self.bkg['isSignal'] = np.zeros(len(self.bkg))

        print 'Training Statistics ----'
        print 'No. Signal', self.sig.shape[0]
        print 'No. Backg.', self.bkg.shape[0]

    def buildTrainingSet(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        ## make the combined dataset
        fat_panda = pd.concat([self.sig, self.bkg])
        self.data = fat_panda.values ## convert pandas dataframe -> numpy array

        # split data into labels and also split into train/test
        data_train, data_test, meta_train, meta_test = train_test_split(
            self.data[:, :self.ninp], self.data[:, self.ninp:], test_size=0.05, random_state=7)

        # normalize all input variables to improve performance
        data_train = StandardScaler().fit_transform(data_train)

        ## separate the event weights and labels
        self.data = data_train
        self.weight = meta_train[:, 0]
        self.label = meta_train[:, 1]
        return data_test, meta_test

    def trainModel(self):
        return self.model.fit(self.data, self.label, shuffle=True,
                      epochs=10000, batch_size=1024, verbose=True,
                      callbacks=self.callbacks, validation_split=0.25, sample_weight=self.weight
                      )


def main(args):
    ## build the network
    cl = Classifier(args.model_name, len(args.vars), args.nhid)

    ## load signal and background data
    cl.loadData(args.vars, args.signal, args.background)

    ## build the training dataset
    data_test, meta_test = cl.buildTrainingSet()

    ## train the model and return info from training
    history = cl.trainModel()

    ## make some pretty plots
    ROC_curve(np.concatenate((data_test, cl.data), axis=0),
              np.concatenate((meta_test[:, 1], cl.label), axis=0),
              np.concatenate((meta_test[:, 0], cl.weight), axis=0),
              cl
              )

    ## extra plots
    if args.verbose:
        trainingPlots(history, cl)
        discPlot(cl, cl.sig, cl.bkg)

    ## save network info to be loaded when running
    if args.model_name != None:
        model_name = args.model_name
    else:
        model_name = 'NN_model'

    create_json(model_name, args.nhid, args.vars)


## Just read all of the CL arguments then run the main function
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description="Train two-layer neural network to separate Drell-Yan + jets from VBF Higgs to tau-tau"
    )
    parser.add_argument('--verbose', action='store_true',
                        dest='verbose', default=False,
                        help='run in verbose mode'
                        )
    parser.add_argument('--nhid', '-n', nargs='+', action='store',
                        dest='nhid', default=[7], type=int,
                        help='[# hidden, ...# nodes in layer]'
                        )
    parser.add_argument('--vars', '-v', nargs='+', action='store',
                        dest='vars', default=['Q2V1', 'Q2V2'],
                        help='variables to input to network'
                        )
    parser.add_argument('--model_name', '-m', action='store',
                        dest='model_name', default=None,
                        help='name of a trained model'
                        )
    parser.add_argument('--signal', action='store',
                        dest='signal', default='input_files/VBF125.root',
                        help='name of the signal file'
                        )
    parser.add_argument('--background', action='store',
                        dest='background', default='input_files/embed.root',
                        help='name of background file'
                        )
    main(parser.parse_args())
