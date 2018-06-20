from Cifar10_Ensemble.Ensemble import Ensemble
x=Ensemble('./Cifar10_Ensemble/cfg_dir/ensemble_0.cfg')
x.load_classifiers()
x.print_all_accs()

'''
innovationcommons@icvr1:~/InnovCommon_Projects/Shakkotai$ python test_ensemble.py 
Loaded Config Info For RandomForest - RFC_0.cfg
/home/innovationcommons/.local/lib/python2.7/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
  from ._conv import register_converters as _register_converters
Using Theano backend.
Using cuDNN version 7005 on context None
Mapped name None to device cuda0: GeForce GTX 1080 (0000:01:00.0)
Loaded Config Info For ResNetV1 - RNV1_0.cfg
Loaded Config Info For ResNetV2 - RNV2_0.cfg
Loaded Config Info For SimpleCNN - SCNN_0.cfg
Loaded Config Info For XGBoost - XGBC_0.cfg


Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/SCNN_0.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/RFC_0_test.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/XGBC_0_test.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/RNV1_0.pkl
Read  /home/innovationcommons/InnovCommon_Projects/Shakkotai/Cifar10_Ensemble/Cifar10_Classifiers/results_dir/RNV2_0.pkl


SimpleCNN - Classifier 0:
Training Acc: 0.4789
Testing Acc: 0.4688


RandomForest - Classifier 0:
Training Acc: 0.9998
Testing Acc: 0.3593


XGBoost - Classifier 0:
Training Acc: 0.4969
Testing Acc: 0.3736


ResNetV1 - Classifier 0:
Training Acc: 0.5043
Testing Acc: 0.4962


ResNetV2 - Classifier 0:
Training Acc: 0.4403
Testing Acc: 0.4255

'''
