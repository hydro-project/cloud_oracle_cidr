from enum import Enum
from typing import Dict, List, Any
import torch
import cvxpy as cp
import numpy as np

from cloud_oracle_prototype.dummy_data import load_linear_functions
from cloud_oracle_prototype.queries.minimization_query import init_query, minimization_query
from cloud_oracle_prototype.queries.minimization_ilp import init_ilp, minimize_ilp
from cloud_oracle_prototype.experiments.experiment import Experiment
from cloud_oracle_prototype.experiments.util import compute_combinations

class ApproachType(Enum):
    ORACLE = "Oracle"
    ILP = "ILP"

class MinimizationExperiment(Experiment):
    
    def __init__(self, *, output_filename, num_warmups=1, verbose=0, experiment_args):
        super().__init__(output_filename=output_filename, num_warmups=num_warmups, verbose=verbose, experiment_args=experiment_args)

        self.approach = None

    def __str__(self):
        return self.pretty_str()
        
    def __repr__(self):
        return self.__repr__()

    def get_benchmark_args(self):
        return self.experiment_args
    
    def get_name(self):
        return f"Minimization Experiment"
    
    def load(self, *, approach: ApproachType, **kwargs):

        if approach == ApproachType.ORACLE:
            return self._load_oracle(**kwargs)
        elif approach == ApproachType.ILP:
            return self._load_ilp(**kwargs)
        else:
            raise ValueError(f"Unknown approach {approach}")
        
    def _load_oracle(self,*, threads, batch_size, device, dtype, num_data_centers, num_client_data_centers, **kwargs):

        # Load data
        num_params = num_client_data_centers * 2
        num_functions = compute_combinations(num_data_centers, 2)

        if self.verbose > 0:
            print(f"Loading {num_functions} functions with {num_params} parameters for {num_data_centers} data centers and {num_client_data_centers} client data centers")
        data = load_linear_functions(device=device, dtype=dtype, num_functions=num_functions, num_params=num_params, as_planes=False, **kwargs)

        # Set number of threads
        torch.set_num_threads(threads)

        # Initialize query state
        (outputs, output_indexes, inputs)= init_query(planes=data, batch_size=batch_size, num_functions=num_functions, num_params=num_params, device=device, dtype=dtype)

        inputs.zero_()
        
        # Create workload lambda
        workload = lambda _: minimization_query(planes=data, inputs=inputs, outputs=outputs, outputs_index=output_indexes)

        return (workload, batch_size, device)

    def _load_ilp(self, *, threads: int, batch_size: int, num_data_centers: int, num_client_data_centers: int, distance_constraint: float = 200.0, device, seed=42, solver=cp.GUROBI):

        # Load the problem
        problem = init_ilp(num_data_centers=num_data_centers, num_client_data_centers=num_client_data_centers, distance_constraint=distance_constraint, seed=seed)
        
        # Function to solve the problem with different read/write frequencies
        if not solver in cp.installed_solvers():
            print(f"Solver {solver} not installed. Falling back to automatic choice by CVXPY.")
            solver = None


        read_frequencies_shortcircuit = np.random.rand(num_client_data_centers)
        write_frequencies_shortcircuit = np.random.rand(num_client_data_centers)

        # Warmup problem compilation
        if self.verbose > 0:
            print("Compiling problem...")
        minimize_ilp(problem=problem, read_frequencies=read_frequencies_shortcircuit, write_frequencies=write_frequencies_shortcircuit, solver=solver, verbose=self.verbose > 1, threads=threads)

        # Create workload lambda
        workload = lambda _: minimize_ilp(problem=problem, read_frequencies=read_frequencies_shortcircuit, write_frequencies=write_frequencies_shortcircuit, solver=solver, verbose=self.verbose > 1, threads=threads)

        return (workload, 1, "cpu")