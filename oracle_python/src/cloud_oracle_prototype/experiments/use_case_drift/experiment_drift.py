from typing import Dict, List, Any
import torch

from cloud_oracle_prototype.dummy_data import load_linear_functions
from cloud_oracle_prototype.queries.directed_drift_query import directed_drift_query
from cloud_oracle_prototype.queries.conservative_drift_query import conservative_drift_query
from cloud_oracle_prototype.experiments.experiment import Experiment
from cloud_oracle_prototype.experiments.util import compute_combinations
from enum import Enum


class DriftType(Enum):
    DIRECTED = "directed"
    UNDIRECTED = "undirected"


class DriftExperiment(Experiment):

    def __init__(self, *, output_filename, num_warmups=1, verbose=0, experiment_args):
        super().__init__(output_filename=output_filename, num_warmups=num_warmups, verbose=verbose, experiment_args=experiment_args)

    def __str__(self):
        return self.pretty_str()
        
    def __repr__(self):
        return self.__repr__()

    def get_benchmark_args(self):
        return self.experiment_args
    
    def get_name(self):
        return f"Drift Experiment"
    
    def load(self, *, drift_type: DriftType, threads, device, dtype, num_data_centers, num_client_data_centers, **kwargs):

        # Load data
        batch_size = 1
        num_params = num_client_data_centers * 2
        num_functions = compute_combinations(num_data_centers, 2)

        data = load_linear_functions(device=device, dtype=dtype, num_functions=num_functions, num_params=num_params, as_planes=True, **kwargs)

        current_parameter = torch.zeros((num_params+1,), dtype=dtype, device=device)

        # Set number of threads
        torch.set_num_threads(threads)
        
        # Create workload lambda

        if drift_type == DriftType.DIRECTED:
            # Also allocate drift
            drift = torch.ones(num_params+1, dtype=dtype, device=device)
            # Create workload lambda
            workload = lambda _: directed_drift_query(current_parameter=current_parameter, planes=data, drift=drift)
        elif drift_type == DriftType.UNDIRECTED:
            workload = lambda _: conservative_drift_query(current_parameter=current_parameter, planes=data)
        else:
            raise ValueError(f"Unknown drift type {drift_type}")

        return (workload, batch_size, device)