from enum import Enum
from typing import Dict, List, Any
import torch

from cloud_oracle_prototype.dummy_data import load_planes_with_savings
from cloud_oracle_prototype.queries.simulation_query import init_query, simulation_query
from cloud_oracle_prototype.experiments.experiment import Experiment
from cloud_oracle_prototype.experiments.util import compute_combinations

class SimulationExperiment(Experiment):
    
    def __init__(self, *, output_filename, num_warmups=1, verbose=0, experiment_args):
        super().__init__(output_filename=output_filename, num_warmups=num_warmups, verbose=verbose, keep_results=True, experiment_args=experiment_args)

    def __str__(self):
        return self.pretty_str()
        
    def __repr__(self):
        return self.__repr__()
    
    def get_benchmark_args(self):
        return self.experiment_args
    
    def get_name(self):
        return f"Simulation Experiment"
        
    def load(self,*, threads, batch_size, device, dtype, num_data_centers, num_data_centers_other, num_client_data_centers, percent_data_centers_A: float, percent_data_centers_B: float, percent_change_A: float, percent_change_B: float, seed = 42):

        # Load data
        num_params = num_client_data_centers * 2
        num_functions = compute_combinations(num_data_centers, 2)
        num_functions_other = compute_combinations(num_data_centers_other, 2)
        
        planes, planes_other = load_planes_with_savings(num_functions=num_functions, num_functions_other=num_functions_other, num_params=num_params, device=device, dtype=dtype, percent_data_centers_A=percent_data_centers_A, percent_data_centers_B=percent_data_centers_B, percent_change_A=percent_change_A, percent_change_B=percent_change_B)

        # Set number of threads
        torch.set_num_threads(threads)

        # Initialize query state
        (outputs, inputs)= init_query(planes=planes, planes_other=planes_other, batch_size=batch_size, num_params=num_params, device=device, dtype=dtype)

        def simulate():
            return simulation_query(planes=planes, planes_other=planes_other, inputs=inputs, outputs=outputs)
        
        # Create workload lambda
        workload = lambda _: simulate()

        return (workload, batch_size, device)