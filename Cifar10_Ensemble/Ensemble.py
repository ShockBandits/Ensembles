from collections import defaultdict
import cPickle as pickle
from configobj import ConfigObj, ConfigObjError, flatten_errors
import importlib
import numpy as np
import os
import pandas as pd
import sys
from validate import Validator

#from .Cifar10_Accessor.readCifar10 import getCifar10, getMetaDict

#from .Cifar10_Classifiers.RandomForest import RandomForest_Classifier
#from .Cifar10_Classifiers.XGBoost import XGBoost_Classifier
#from .Cifar10_Classifiers.SimpleCNN import SimpleCNN_Classifier
#from .Cifar10_Classifiers.ResNetV1 import ResNetV1_Classifier
#from .Cifar10_Classifiers.ResNetV2 import ResNetV2_Classifier

class Ensemble(object):

    def __init__(self, cfg_file):

        try:
            config = ConfigObj(cfg_file, file_error = True)
        except (ConfigObjError, IOError), e:
            print "\n\nCouldn't read '%s' : %s\n\n"%(cfg_file, e)
            sys.exit(1)
            
        self.classifier_dir = config['classifier_root_dir']
        self.classifier_dict = defaultdict(list)

        clfr_dict = config['classifiers']
        for curr_clfr in clfr_dict:
            c_module = importlib.import_module('Cifar10_Ensemble.Cifar10_Classifiers.'+curr_clfr)
            constructor = getattr(c_module, curr_clfr+'_Classifier', None)
            
            
            if isinstance(clfr_dict[curr_clfr], str):
                self.classifier_dict[curr_clfr].append(constructor(clfr_dict[curr_clfr]))
                print "Loaded Config Info For %s - %s"%(curr_clfr,
                                                        clfr_dict[curr_clfr])
            elif isinstance(clfr_dict[curr_clfr], list):
                for clfr_elem in clfr_dict[curr_clfr]:
                    self.classifier_dict[curr_clfr].append(constructor(clfr_elem))
                    print "Loaded Config Info For %s - %s"%(curr_clfr,
                                                            clfr_elem)
            else:
                print "\n\nCan't Read Config Info For %s -  %s\n\n"%(curr_clfr,
                                                                     clfr_dict[curr_clfr])
                sys.exit(1)
        print "\n"


    def load_classifiers(self):

        for curr_clfrs in self.classifier_dict:
            for curr in self.classifier_dict[curr_clfrs]:
                curr.read()
        print "\n"


    def print_indiv_acc(self, classifier_type, classifier_num = 0):
        classifier = self.classifier_dict[classifier_type][classifier_num]
        print "%s - Classifier %s:"%(classifier_type, classifier_num)
        classifier.print_acc()

    def print_all_accs(self):
        for curr_clfr_type in self.classifier_dict:
            for ctr, clfr in enumerate(self.classifier_dict[curr_clfr_type]):
                self.print_indiv_acc(curr_clfr_type, ctr)
                print 
            print 
        print "\n"
            

        
                                                        
