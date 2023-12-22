import unittest
from oracle.queries.minimization_ilp import init_ilp, minimize_ilp
import numpy as np

class TestMinimizationILP(unittest.TestCase):
    def test_minimization_ilp(self):
        # Initialize the problem
        problem = init_ilp(num_data_centers=3, num_client_data_centers=2, distance_constraint=0.5)

        # Solve the problem
        res = minimize_ilp(problem=problem, read_frequencies=np.array([0.5, 0.5]), write_frequencies=np.array([0.5, 0.5]), threads=1, solver=None, verbose=True)

        # Print the result
        print(res)

if __name__ == '__main__':
    unittest.main()