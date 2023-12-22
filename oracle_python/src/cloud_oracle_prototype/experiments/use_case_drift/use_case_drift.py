import os
from typing import List
import torch
from cloud_oracle_prototype.experiments.use_case_drift.experiment_drift import DriftExperiment, DriftType

shared_arguments = {
    "num_iterations": [100],
    "threads": [40],
    "device": ["cuda", "cpu"],
    "drift_type": [DriftType.DIRECTED, DriftType.UNDIRECTED],
    "dtype": [torch.float16],
}

arguments_data_center_scaling = {
    "num_data_centers": [2, 10, 100, 300, 10**3],
    "num_client_data_centers": [300],
}

arguments_client_data_center_scaling = {
    "num_data_centers": [300],
    "num_client_data_centers": [2, 10, 100, 300, 10**3],
}

def generate_experiments(*, output_dir: str, verbose: int = 1, num_warmups: int = 1,) -> List[DriftExperiment]:
    output_dir_use_case_1 = os.path.join(output_dir, "use_case_drift")
    os.makedirs(output_dir_use_case_1, exist_ok=True)

    experiments = []

    output_filename_data_center_scaling = os.path.join(output_dir_use_case_1, "data_center_scaling_oracle.csv")
    args = {**shared_arguments, **arguments_data_center_scaling}
    data_center_scaling_experiment = DriftExperiment(output_filename=output_filename_data_center_scaling, num_warmups=num_warmups, verbose=verbose, experiment_args=args)
    experiments.append(data_center_scaling_experiment)

    output_filename_client_data_center_scaling = os.path.join(output_dir_use_case_1, "client_data_center_scaling_oracle.csv")
    args = {**shared_arguments, **arguments_client_data_center_scaling}
    client_data_center_scaling_experiment = DriftExperiment(output_filename=output_filename_client_data_center_scaling, num_warmups=num_warmups, verbose=verbose, experiment_args=args)
    experiments.append(client_data_center_scaling_experiment)

    return experiments