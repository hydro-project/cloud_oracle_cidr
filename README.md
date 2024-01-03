# Optimizing the cloud? Don’t train models. Build oracles!

This repository contains the prototype of `Cloud Oracles` proposed in CIDR'24.
See below for implementation and experiment details for precomputing and querying cloud oracles.

```
@inproceedings{cidr_2024/cloud_oracle,
  author       = {Tiemo Bang and Conor Power and Siavash Ameli and Natacha Crooks and Joseph M. Hellerstein},
  title        = {Optimizing the cloud? Don’t train models. Build oracles!},
  booktitle    = {14th Annual Conference on Innovative Data Systems Research, {CIDR} 2024 Chaminade, USA, January 14-17, 2024},
  publisher    = {{www.cidrdb.org}},
  year         = {2024},
  url          = {{https://www.cidrdb.org/cidr2024/papers/p47-bang.pdf}}
}
```

# Precomputing the Cloud Oracle

The Hydroflow program in the [precomputation](./precomputation/) directory demonstrates how to precompute Cloud Oracles for fault-tolerant object placement.

Specifically, this program implements the enumeration and filtering logic of [§3.1](https://www.cidrdb.org/cidr2024/papers/p47-bang.pdf#page=3) for: Placement of an object in 2 data centers 200km apart that minimizes the latency for writing to both data centers and reading from either.

Note that, this prototype uses synthetic data!

There are several variants.
[`Monolith - Enumeration Only`](./precomputation/src/monolith_enumeration.rs) implements the logic of enumerating placement decisions and applying the distance constraint.
[`Monolith`](./precomputation/src/monolith.rs) additionally filters the enumerated decision by computing which decisions have dominated access latency and hence are never a good choice. Both these variants are monolithic in that they yield highly optimized but single-threaded executables. A parallel variant, which will execute the precomputation in data-parallel and pipelined stages, is pending -- see `parallel` branch.

## Setup

Install the rust, e.g., via [rustup](https://www.rust-lang.org/tools/install) (for Linux/Mac:`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`).

## Run the experiment

```
cd precomputation
cargo run -r -- --output_dir=../results
```

This command compiles Hydroflow program in release mode, executes the experiment, and write the result in to [results/precomputation/precomputation.csv](results/precomputation/precomputation.csv).

The experiment computes a number of Cloud Oracles for different numbers of data centers hosting object stores (`num_d`) and data centers hosting client applications that access the objects (`num_c`).
These parameters are hard-coded in [main.rs](precomputation/src/main.rs) according to the experiments in the CIDR publication.

# Querying the Cloud Oracle

The python package in [`oracle_python`](./oracle_python/) implements several online optimization tasks of cloud oracles: scenario planning via simulation, and migration planning under workload drift.

The query logic on top of the precomputed look-up structure is implemented in [`queries`](./oracle_python/src/cloud_oracle_prototype/queries/) as specified in [§3.2](https://www.cidrdb.org/cidr2024/papers/p47-bang.pdf#page=4), [§3.3](https://www.cidrdb.org/cidr2024/papers/p47-bang.pdf#page=5), [§3.4](https://www.cidrdb.org/cidr2024/papers/p47-bang.pdf#page=6) of the paper, respectively.
The experiments in [experiment](./oracle_python/src/cloud_oracle_prototype/experiments/) are specified likewise.
In the subdirectories, you will find for each use case the experiment definitions in `use_case_X.py` and the setup logic in `experiment_X.py`.

Note that, the experiments use synthetic data!

## Setup

```
cd oracle_python
python -m pip install .
```

## Run the experiments

```
python -m cloud_oracle_prototype --all --output-dir results
```

This executes all experiments of the CIDR publication. Alternatively, experiments can be run individually, see `python -m cloud_oracle_prototype --help`.
The GPU- or MPS- based experiments are only executed if the according hardware is detected.

# Plotting results

See [results.ipynb](./results.ipynb) for plotting. It assumes all result are stored the in [results directory](./results/) and according subdirectories of the experiments.

## Setup

Assuming the basic infrastructure for Notebooks is present, install the dependencies for plotting:

```
python -m pip install -r requirements.txt
```
