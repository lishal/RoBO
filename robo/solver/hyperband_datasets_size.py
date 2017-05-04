import os
import time
import json
import logging
from robo.solver.base_solver import BaseSolver

import numpy as np

try:
    import multibeep as mb

except ImportError as e:
    raise ValueError("If you want to use Hyperband you have to install the following dependencies:\n"
                     "multibeep (see https://github.com/automl/multibeep)")


logger = logging.getLogger(__name__)


class hyperband_arm(mb.arms.python):
    def __init__(self, task, configuration, subset_fractions):
        self.task = task
        self.configuration = configuration
        self.subset_fractions = subset_fractions
        self.costs = []
        self.i = 0
        super().__init__(self, b"Hyperband arm wrapper")

    def pull(self):
        if self.i == len(self.subset_fractions):
            raise "Ooops, that shouldn't happen. Trying to pull this arm too many times."
        onyx = self.task.objective_function(self.configuration,
                                            dataset_fraction=self.subset_fractions[self.i])
        self.i += 1
        self.costs.append(onyx['cost'])
        return -onyx['function_value']
    # rest of the methods don't have to be specified here


class HyperBand_DataSubsets(BaseSolver):
    """
    variables to use the save_iteration function of the BaseSolver class:


    """
    def __init__(self, task, eta, min_subset_fraction, output_path=None, rng=None):
        """
        Parameters
        ----------

        task : hpolib.benchmark.AbstractBenchmark object
            the task should accept dataset_fraction argument (between 0 and 1).
        eta : float
            In each iteration, a complete run of sequential halving is executed. In it,
            after evaluating each configuration on the same subset size, only a fraction of
            1/eta of them 'advances' to the next round.
            Must be greater or equal to 2.
        min_subset_fraction : float
            size of the smallest subset to consider. The sizes will be
            geometrically distributed $\sim \eta^k$ for
            $k\in [0, 1, ... , num_subsets - 1]$ where
            $\eta^{num_subsets - 1} \geq min_subset_fraction$
        output_path: string
            Specifies the path where the intermediate output after each iteration will be saved.
            If None no output will be saved to disk.
        rng: numpy.random.RandomState
        """

        self.task = task
        self.eta = eta
        self.min_subset_fraction = min_subset_fraction
        self.output_path = output_path

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng
        #TODO: set seed of the configuration space

        task.configuration_space.seed(self.rng.randint(np.iinfo(np.int16).max))

        self.X = []
        self.Y = []
        self.incumbent = None
        self.incumbent_value = float('inf')
        self.incumbents = []
        self.incumbent_values = []
        self.time_func_eval = []
        self.runtime = []
        self.time_start = None

    def run(self, num_iterations=10, X=None, Y=None, overwrite=False):
        """
        The main optimization loop

        Parameters
        ----------
        num_iterations: int
            The number of iterations
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        Y: np.ndarray(N,1)
            Function values of the already evaluated points

        Returns
        -------
        np.ndarray(1,D)
            Incumbent
        np.ndarray(1,1)
            (Estimated) function value of the incumbent
        """
        self.time_start = time.time()

        eta = self.eta
        num_subsets = -int(np.log(self.min_subset_fraction)/np.log(eta)) + 1
        subset_fractions = np.power(eta, -np.linspace(num_subsets-1, 0, num_subsets))

        for it in range(num_iterations):
            logger.info("Start iteration %d ... ", it)

            # compute the the value of s for this iteration
            s = num_subsets - 1 - (it % num_subsets)

            # the number of initial configurations
            n = int(np.floor((num_subsets)/(s+1)) * eta**s)

            # set up the arms with random configurations
            configurations = [self.choose_next() for i in range(n)]
            arms = [hyperband_arm( self.task, c,
                                   subset_fractions[(-s-1):]) for c in configurations]

            # set up the bandit and the policy and play
            bandit = mb.bandits.last_n_pulls(n=1)
            [bandit.add_arm(a) for a in arms]

            policy = mb.policies.successive_halving(
                bandit, 1, eta, factor_pulls = 1)
            for _ in range(s+1):
                policy.play_n_rounds(1)

                # the best configuration is the first arm
                best_config_index = bandit[0].identifier

                c = configurations[best_config_index]
                v = - bandit[0].estimated_mean

                if v < self.incumbent_value:
                    self.incumbent = c
                    self.incumbent_value = v

                # book keeping
                self.incumbents.append(self.incumbent)
                self.incumbent_values.append(self.incumbent_value)
                self.time_func_eval.append(sum([sum(a.costs) for a in arms]))
                self.runtime.append(time.time() - self.time_start)
                if self.output_path is not None:
                    self.save_output()

            for i in range(len(arms)):
                if len(arms[bandit[i].identifier].costs) == bandit[0].num_pulls:
                    self.X.append(arms[bandit[i].identifier].configuration)
                    self.Y.append(bandit[i].estimated_mean)


    def choose_next(self, X=None, Y=None):
        """
        Suggests a new point to evaluate.

        Parameters
        ----------
        X: np.ndarray(N,D)
            Initial points that are already evaluated
        Y: np.ndarray(N,1)
            Function values of the already evaluated points

        Returns
        -------
        ConfigSpace.Configuration
            Suggested point
        """
        return self.task.configuration_space.sample_configuration()

    def save_output(self):
        data = dict()
        data["runtime"] = self.runtime
        # Note that the ConfigSpace automatically converts to the [0, 1]^D space
        data["incumbent"] = [t.get_array().tolist() for t in self.incumbents] 
        data["incumbents_value"] = self.incumbent_values
        data["time_func_eval"] = self.time_func_eval

        json.dump(data, open(os.path.join(self.output_path, "hyperband_results.json"), "w"))
