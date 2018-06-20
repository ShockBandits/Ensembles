from Cifar10_Ensemble.Ensemble import Ensemble
x=Ensemble('./Cifar10_Ensemble/cfg_dir/ensemble_0.cfg')
x.load_classifiers()
x.print_all_accs()
