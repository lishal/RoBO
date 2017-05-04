import os
import sys
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

from robo.solver.hyperband_datasets_size import HyperBand_DataSubsets

from hpolib.benchmarks.ml.svm_benchmark import SvmOnMnist, SvmOnVehicle, SvmOnCovertype
#from hpolib.benchmarks.ml.residual_networks import ResidualNeuralNetworkOnCIFAR10
from hpolib.benchmarks.ml.conv_net import ConvolutionalNeuralNetworkOnCIFAR10
#, ConvolutionalNeuralNetworkOnSVHN


run_id = int(sys.argv[1])
dataset = sys.argv[2]
seed = int(sys.argv[3])

rng = np.random.RandomState(seed)
if dataset == "mnist":
    f = SvmOnMnist(rng=rng)
    output_path = "./experiments/fabolas/results/svm_%s/hyperband_%d" % (dataset, run_id)
    s_max = f.train.shape[0]
    s_min = 100
elif dataset == "vehicle":
    f = SvmOnVehicle(rng=rng)
    output_path = "./experiments/fabolas/results/svm_%s/hyperband_%d" % (dataset, run_id)
    s_max = f.train.shape[0]
    s_min = 100
elif dataset == "covertype":
    f = SvmOnCovertype(rng=rng)
    output_path = "./experiments/fabolas/results/svm_%s/hyperband_%d" % (dataset, run_id)
    s_max = f.train.shape[0]
    s_min = 100
elif dataset == "cifar10":
    f = ConvolutionalNeuralNetworkOnCIFAR10(rng=rng)
    output_path = "./experiments/fabolas/results/cnn_%s/hyperband_%d" % (dataset, run_id)
    s_max = f.train.shape[0]
    s_min = 512 
elif dataset == "svhn":
    f = ConvolutionalNeuralNetworkOnSVHN(rng=rng)
    output_path = "./experiments/fabolas/results/cnn_%s/hyperband_%d" % (dataset, run_id)
#elif dataset == "res_net":
#    f = ResidualNeuralNetworkOnCIFAR10(rng=rng)
#    output_path = "./experiments/fabolas/results/res_%s/hyperband_%d" % (dataset, run_id)


os.makedirs(output_path, exist_ok=True)

if 'loggingInitialized' not in locals():
    loggingInitialized = True

    path = output_path+'/RoBOlog.txt'
    logging.basicConfig(level=logging.DEBUG,filename=path)
eta = 4.
B = -int(np.log(s_min/s_max)/np.log(eta))+1

print(B)

opt = HyperBand_DataSubsets(f, eta, eta**(-(B-1)), output_path=output_path, rng=rng)

opt.run(int(20 / B))

test_error = []
for c in opt.incumbents:
    test_error.append(f.objective_function_test(c)["function_value"])

    results = dict()
    results["incumbents"] = opt.incumbents
    results["test_error"] = test_error
    results["runtime"] = opt.runtime
    results["time_func_eval"] = opt.time_func_eval
    results["run_id"] = run_id

    with open(os.path.join(output_path, 'results_%d.json' % run_id), 'w') as fh:
        json.dump(results, fh)
