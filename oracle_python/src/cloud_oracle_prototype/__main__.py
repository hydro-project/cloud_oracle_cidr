import argparse


def main():
    parser = argparse.ArgumentParser(prog="cloud_oracle_prototype", description="Prototype of a cloud oracle")
    parser.add_argument("--output-dir", required=True, help="Base output directory")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--all", action="store_true", help="Run all experiments. Alternatively use: --minimization --simulation --drift")
    parser.add_argument("--dry-run", action="store_true", help="Do not run experiments, only print them")
    
    args_use_case_1 = parser.add_argument_group("Use case: Minimization", "Configure and run minimization experiments, i.e., minimizing access latency under different numbers of data centers and client data centers.")
    args_use_case_1.add_argument("--minimization", action="store_true", help="Run minimization experiments")
    args_use_case_1.add_argument("--do-oracle", action="store_true", help="Run experiments with the oracle")
    args_use_case_1.add_argument("--do-ilp", action="store_true", help="Run experiments with the ILP")
    args_use_case_1.add_argument("--do-data-center-scaling", action="store_true", help="Run experiments with different numbers of data centers")
    args_use_case_1.add_argument("--do-client-data-center-scaling", action="store_true", help="Run experiments with different numbers of client data centers")

    args_simulation = parser.add_argument_group("Use case: Simulation", "Configure and run simulation experiments, i.e., simulating the improvement in access latency with increasing number of samples.")
    args_simulation.add_argument("--simulation", action="store_true", help="Run simulation experiments")

    args_drift = parser.add_argument_group("Use case: Drift", "Configure and run drift experiments, i.e., minimizing access latency for drifting paramters under different numbers of data centers and client data centers.")
    args_drift.add_argument("--drift", action="store_true", help="Run drift experiments")

    args = parser.parse_args()

    base_output_dir = args.output_dir

    experiments = []
    if args.all or args.minimization:
        from cloud_oracle_prototype.experiments.use_case_minimization import use_case_minimization
        experiments.extend(use_case_minimization.generate_experiments(output_dir=base_output_dir, do_ilp=args.do_ilp or args.all, do_oracle=args.do_oracle or args.all, do_data_center_scaling=args.do_data_center_scaling or args.all, do_client_data_center_scaling=args.do_client_data_center_scaling or args.all, verbose=args.verbose))

    if args.all or args.simulation:
        from cloud_oracle_prototype.experiments.use_case_simulation import use_case_simulation
        experiments.extend(use_case_simulation.generate_experiments(output_dir=base_output_dir, verbose=args.verbose))

    if args.all or args.drift:
        from cloud_oracle_prototype.experiments.use_case_drift import use_case_drift
        experiments.extend(use_case_drift.generate_experiments(output_dir=base_output_dir, verbose=args.verbose))

    print(f"Running {len(experiments)} experiments")
    if args.verbose > 1 or args.dry_run:
        for experiment in experiments:
            print(experiment)

    if not args.dry_run:
        for experiment in experiments:
            experiment.benchmark()

if __name__ == "__main__":
    main()