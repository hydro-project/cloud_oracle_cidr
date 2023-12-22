import numpy as np
import cvxpy as cp

def init_ilp(*, num_data_centers: int, num_client_data_centers: int, distance_constraint: float, skip_distance = True, seed=42, verbose: int = 0):

    if verbose > 0:
        print(f"Initializing ILP with {num_data_centers} data centers, {num_client_data_centers} client data centers, and distance constraint {distance_constraint}")

    # Seed the random number generator for reproducibility
    np.random.seed(seed)

    # l: Matrix of latencies between client data centers and data centers (num_client_data_centers, num_data_centers)
    latencies = np.random.rand(num_client_data_centers, num_data_centers)

    # Parameters for read and write frequency coefficients
    read_frequencies_param = cp.Parameter(num_client_data_centers, nonneg=True, name="read_frequencies")
    write_frequencies_param = cp.Parameter(num_client_data_centers, nonneg=True, name="write_frequencies")

    # dist(d, d'): Matrix of distances between data centers (num_data_centers, num_data_centers)
    distances = np.full((num_data_centers, num_data_centers), distance_constraint + 1)
    np.fill_diagonal(distances, 0)  # Distance to itself is 0

    # Define Variables
    x = cp.Variable((num_data_centers,), boolean=True, name="x")  # Eq. (13)
    y = cp.Variable((num_client_data_centers, num_data_centers), boolean=True, name="y")  # Eq. (13)
    if not skip_distance:
        v = cp.Variable((num_data_centers, num_data_centers), boolean=True, name="v")  # Eq. (14)
    z = cp.Variable(num_client_data_centers, name="z")  # Eq. (15)

    # Objective Function (Eq. (5))
    raw_read_latency = cp.multiply(latencies, y)
    weighted_read_frequency = raw_read_latency.T @ read_frequencies_param
    objective = cp.Minimize(cp.sum(z) + cp.sum(weighted_read_frequency))

    # Constraints
    constraints = []

    if not skip_distance:
        # Distance Constraints (Eq. (6))
        for d in range(num_data_centers):
            for dp in range(num_data_centers):
                if d != dp:
                    constraints.append(distance_constraint * v[d, dp] <= distances[d, dp])

        # Auxiliary Mapping of v to x (Eqs. (7) and (8))
        for d in range(num_data_centers):
            for dp in range(num_data_centers):
                if d != dp:
                    constraints.append(2 * v[d, dp] <= x[d] + x[dp])
                    constraints.append(v[d, dp] >= x[d] + x[dp] - 1)

    # Write and Read Latency Constraints (Eq. (9))
    for c in range(num_client_data_centers):
        for d in range(num_data_centers):
            constraints.append(z[c] >= latencies[c, d] * write_frequencies_param[c] * x[d])

    # Ensuring exactly two data centers are selected for writing, one for reading (Eqs. (10) and (11))
    constraints.append(cp.sum(x) == 2)
    for c in range(num_client_data_centers):
        constraints.append(cp.sum(y[c, :]) == 1)

    # Read only if writing (Eq. (12))
    for c in range(num_client_data_centers):
        for d in range(num_data_centers):
            constraints.append(x[d] >= y[c, d])

    # Problem
    problem = cp.Problem(objective, constraints)

    return problem

def minimize_ilp(*, problem: cp.Problem, read_frequencies, write_frequencies, threads, solver, verbose):
            
    # Get the parameters of the problem
    read_frequencies_param = problem.param_dict["read_frequencies"]
    write_frequencies_param = problem.param_dict["write_frequencies"]
    # Set the parameter values
    read_frequencies_param.value = read_frequencies
    write_frequencies_param.value = write_frequencies

    if verbose:
        # Print problem formulation
        print("Problem Formulation:")
        print(problem)

    # Solve the problem
    res = problem.solve(threads=threads, verbose=verbose, solver=solver)

    # Assert that the problem was solved, i.e., is a float
    assert isinstance(res, float), f"Problem could not be solved. Return value: {res}"

    # Extract the solution
    return res