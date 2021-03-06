import os
import sys
import json
import logging
import numpy as np
import time

logging.basicConfig(level=logging.INFO)

from robo.fmin import fabolas

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
    num_iterations = 80
    output_path = "./experiments/fabolas/results/svm_%s/fabolas_%d" % (dataset, run_id)
    s_max = f.train.shape[0]
    print("training set size: ",s_max)
    s_min = 100  # 10 * number of classes
    subsets = [64., 32, 16, 8]

elif dataset == "vehicle":
    f = SvmOnVehicle(rng=rng)
    num_iterations = 80
    output_path = "./experiments/fabolas/results/svm_%s/fabolas_%d" % (dataset, run_id)
    s_max = f.train.shape[0]
    print("training set size: ",s_max)
    print("valid set size: ",f.valid.shape[0])
    s_min = 100  # 10 * number of classes
    subsets = [64., 32, 16, 8]

elif dataset == "covertype":
    f = SvmOnCovertype(rng=rng)
    num_iterations = 80
    output_path = "./experiments/fabolas/results/svm_%s/fabolas_%d" % (dataset, run_id)
    s_max = f.train.shape[0] 
    print("training set size: %d",s_max)
    s_min = 100  # 10 * number of classes
    subsets = [64., 32, 16, 8]

elif dataset == "cifar10":
    f = ConvolutionalNeuralNetworkOnCIFAR10(rng=rng)
    num_iterations = 50
    output_path = "./experiments/fabolas/results/cnn_%s/fabolas_%d" % (dataset, run_id)
    s_max = f.train.shape[0]
    s_min = 512  # Maximum batch size
    subsets = [64., 32, 16, 8]

elif dataset == "svhn":
    f = ConvolutionalNeuralNetworkOnSVHNLocal(rng=rng)
    num_iterations = 50
    output_path = "./experiments/fabolas/results/cnn_%s/fabolas_%d" % (dataset, run_id)
    s_max = f.train.shape[0]
    s_min = 512  # Maximum batch size
    subsets = [64., 32, 16, 8]

elif dataset == "res_net":
    f = ResidualNeuralNetworkOnCIFAR10(rng=rng)
    num_iterations = 50
    output_path = "./experiments/fabolas/results/res_%s/fabolas_%d" % (dataset, run_id)
    s_max = f.train.shape[0]
    s_min = 128  # Batch size
    subsets = [256, 128, 64., 32]


os.makedirs(output_path, exist_ok=True)


if 'loggingInitialized' not in locals():
    loggingInitialized = True

    path = output_path+'/RoBOlog.txt'
    logging.basicConfig(level=logging.DEBUG,filename=path)

def objective(x, s):
    dataset_fraction = s / s_max

    res = f.objective_function(x, dataset_fraction=dataset_fraction)
    return res["function_value"], res["cost"]

info = f.get_meta_information()
bounds = np.array(info['bounds'])
lower = bounds[:, 0]
upper = bounds[:, 1]
results = fabolas(objective_function=objective, lower=lower, upper=upper,
                  s_min=s_min, s_max=s_max, n_init=10, num_iterations=1000,
                  n_hypers=30, subsets=subsets,
                  rng=rng, output_path=output_path,run_time = 12*3600)

results["run_id"] = run_id
results['X'] = results['X'].tolist()
results['y'] = results['y'].tolist()
results['c'] = results['c'].tolist()

test_error = []
inc_dict = {}

key = "incumbents"

for inc in results["incumbents"]:
    print(inc)
    if tuple(inc) in inc_dict: 
        test_error.append(inc_dict[tuple(inc)])
    else:
        start_time = time.time()
        y = f.objective_function_test(inc)["function_value"]
        duration = time.time() - start_time
        test_error.append([y,duration])
        inc_dict[tuple(inc)] = [y,duration] 

    results["test_error"] = test_error

    with open(os.path.join(output_path, 'results_%d.json' % run_id), 'w') as fh:
        json.dump(results, fh)
