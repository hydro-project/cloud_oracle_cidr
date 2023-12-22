import os
from typing import List
import torch

from cloud_oracle_prototype.experiments.use_case_simulation.experiment_simulation import SimulationExperiment

shared_arguments = {
    "threads": [40],
    "batch_size": [10**2, 10**3, 10**4], #, 10**5, 10**6],
    "device": ["cuda"], #["cpu"],
    "dtype": [torch.float16],
    "num_data_centers": [300],
    "num_data_centers_other": [300],
    "num_client_data_centers": [300],
    "percent_data_centers_A": [0.3],
    "percent_data_centers_B": [0.3],
    "percent_change_A": [2.0],
    "percent_change_B": [0.1],
    "num_iterations": [1],
}

def generate_experiments(*, output_dir, num_warmups=1, verbose=0) -> List[SimulationExperiment]:
    output_dir_use_case_1 = os.path.join(output_dir, "use_case_simulation")
    os.makedirs(output_dir_use_case_1, exist_ok=True)

    experiments = []

    output_filename = os.path.join(output_dir_use_case_1, "simulation.csv")
    args = {**shared_arguments}
    experiment = SimulationExperiment(output_filename=output_filename, num_warmups=num_warmups, verbose=verbose, experiment_args=args)
    experiments.append(experiment)

    return experiments