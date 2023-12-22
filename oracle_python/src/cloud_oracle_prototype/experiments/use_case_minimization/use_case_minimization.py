import os
from typing import List
import torch
import cvxpy as cp
from cloud_oracle_prototype.experiments.use_case_minimization.experiment_minimization import MinimizationExperiment, ApproachType

shared_arguments = {
    "num_iterations": [100],
    "threads": [40],
    "device": ["cuda", "cpu"],
    "batch_size": [1],
}

arguments_data_center_scaling = {
    "num_data_centers": [2, 10, 100, 300, 10**3],
    "num_client_data_centers": [300],
}

arguments_client_data_center_scaling = {
    "num_data_centers": [300],
    "num_client_data_centers": [2, 10, 100, 300, 10**3],
}

arguments_oracle = {
    "approach": [ApproachType.ORACLE],
    "dtype": [torch.float16],
}

arguments_ilp = {
    "approach": [ApproachType.ILP],
    "solver": [cp.GUROBI],
    "num_iterations": [1],
}

def generate_experiments(*, output_dir: str, verbose: int = 1, num_warmups: int = 1, do_ilp = True, do_oracle = True, do_data_center_scaling = True, do_client_data_center_scaling = True) -> List[MinimizationExperiment]:
    output_dir_use_case_1 = os.path.join(output_dir, "use_case_minimization")
    os.makedirs(output_dir_use_case_1, exist_ok=True)

    experiments = []
    if do_oracle:
        if do_data_center_scaling:
            output_filename_data_center_scaling = os.path.join(output_dir_use_case_1, "data_center_scaling_oracle.csv")
            args = {**shared_arguments, **arguments_data_center_scaling, **arguments_oracle}
            data_center_scaling_experiment = MinimizationExperiment(output_filename=output_filename_data_center_scaling, num_warmups=num_warmups, verbose=verbose, experiment_args=args)
            experiments.append(data_center_scaling_experiment)

        if do_client_data_center_scaling:
            output_filename_client_data_center_scaling = os.path.join(output_dir_use_case_1, "client_data_center_scaling_oracle.csv")
            args = {**shared_arguments, **arguments_client_data_center_scaling, **arguments_oracle}
            client_data_center_scaling_experiment = MinimizationExperiment(output_filename=output_filename_client_data_center_scaling, num_warmups=num_warmups, verbose=verbose, experiment_args=args)
            experiments.append(client_data_center_scaling_experiment)

    if do_ilp:
        if do_data_center_scaling:
            output_filename_data_center_scaling = os.path.join(output_dir_use_case_1, "data_center_scaling_ilp.csv")
            args = {**shared_arguments, **arguments_data_center_scaling, **arguments_ilp}
            data_center_scaling_experiment = MinimizationExperiment(output_filename=output_filename_data_center_scaling, num_warmups=num_warmups, verbose=verbose, experiment_args=args)
            experiments.append(data_center_scaling_experiment)

        if do_client_data_center_scaling:
            output_filename_client_data_center_scaling = os.path.join(output_dir_use_case_1, "client_data_center_scaling_ilp.csv")
            args = {**shared_arguments, **arguments_client_data_center_scaling, **arguments_ilp}
            client_data_center_scaling_experiment = MinimizationExperiment(output_filename=output_filename_client_data_center_scaling, num_warmups=num_warmups, verbose=verbose, experiment_args=args)
            experiments.append(client_data_center_scaling_experiment)

    return experiments