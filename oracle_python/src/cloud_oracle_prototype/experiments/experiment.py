import itertools
import os
from typing import Any, Callable, Dict, List
import pandas as pd
import torch
import time

from abc import ABC, abstractmethod
from tqdm import tqdm

from cloud_oracle_prototype.util import check_device_type_available

class Experiment(ABC):
    num_iterations: int
    output_filename: str
    num_warmups: int
    verbose: int
    keep_results: bool = False
    experiment_args: Dict[str, List[Any]]

    def __init__(self, *, output_filename, num_iterations=10, num_warmups=1, verbose=0, keep_results=False, experiment_args=None):

        self.num_iterations = num_iterations
        self.output_filename = output_filename
        self.num_warmups = num_warmups
        self.verbose = verbose
        self.keep_results = keep_results
        self.experiment_args = experiment_args

        # Filter out unavailable devices
        if self.experiment_args is not None and "device" in self.experiment_args:

            devices_available = []
            for device in self.experiment_args["device"]:
                if check_device_type_available(device):
                    devices_available.append(device)
                else:
                    print(f"Device {device} not available, skipping {self.get_name()}")

            self.experiment_args["device"] = devices_available

        # Create output directory if it does not exist
        output_dir = os.path.dirname(self.output_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def pretty_str(self):
        return f"{self.get_name()}(..., experiment_args={self.get_benchmark_args()})"

    @abstractmethod
    def get_benchmark_args(self) -> Dict[str, List[Any]]:
        """
        Returns a dictionary of arguments to be used for the benchmark.

        The keys of the dictionary are the names of the arguments, the values are lists of values for the arguments.
        The cross product of the values is used to generate the arguments for the benchmark.
        """
        pass
    
    @abstractmethod
    def load(self, **kwargs) -> (Callable, int, str):
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def benchmark(self):

        measurements = []

        # Iterate over cross product of arguments
        benchmark_args = self.get_benchmark_args()
        argument_names = list(benchmark_args.keys())

        for argument_values in tqdm(itertools.product(*benchmark_args.values()), desc=self.get_name()):
            load_kwargs = dict(zip(argument_names, argument_values))
            num_iterations = load_kwargs.pop("num_iterations", self.num_iterations)

            if self.verbose > 0:
                print(f"Running {num_iterations} iterations with arguments: {load_kwargs}")

            # Load the benchmark for arguments
            (workload, batch_size, device) = self.load(**load_kwargs)

            # Warmup
            self._run(workload=workload, num_iterations=self.num_warmups, device=device)

            # Execute
            elapsed_time, results = self._run(workload=workload, num_iterations=num_iterations, device=device)

            time_per_simulation = elapsed_time / num_iterations

            # Store results
            load_kwargs["elapsed_time"] = elapsed_time
            load_kwargs["time_per_simulation"] = time_per_simulation
            if self.keep_results:
                load_kwargs["results"] = results

            measurements.append(load_kwargs)

            self._store_results(measurements)

    def _run(self, *, workload, num_iterations, device):
        if "cuda" in device:
            return self._run_cuda(workload=workload, num_iterations=num_iterations, device=device)
        else:
            return self._run_non_cuda(workload=workload, num_iterations=num_iterations)

    def _run_non_cuda(self, *, workload, num_iterations):
        # Start timing
        start_time = time.time()

        # Run workload
        results = self._run_keep_results(workload=workload, num_iterations=num_iterations)

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time

        return elapsed_time, results

    def _run_cuda(self, *, workload, num_iterations, device):

        with torch.cuda.device(device):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Start timing
            start_event.record()

            # Run workload
            results = self._run_keep_results(workload=workload, num_iterations=num_iterations)

            # End timing
            end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            elapsed_time_s = start_event.elapsed_time(end_event) / 1000.0
            return elapsed_time_s, results
        
    def _run_keep_results(self, *, workload, num_iterations):
        results = []
        if self.keep_results:
            results = [ workload(i) for i in range(num_iterations)]
        else:
            for i in range(num_iterations):
                workload(i)

        return results
        
    def _store_results(self, results):
        # Always overwrite existing results

        # Check if file exists
        res = pd.DataFrame(results)
        res.to_csv(self.output_filename, index=False)
        if self.verbose > 0:
            print(res)
