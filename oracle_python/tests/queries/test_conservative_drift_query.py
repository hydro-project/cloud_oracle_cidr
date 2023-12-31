import unittest
import torch
from oracle.dummy_data import load_linear_functions
from oracle.queries.conservative_drift_query import conservative_drift_query

LOAD_ARGS = dict(
    num_functions = 200,
    num_params = 10,
    coeff_min = 0,
    coeff_max = 1,
    dtype = torch.float16,
    as_planes = True,
    random = False
)

class TestConservativeQuery(unittest.TestCase):

    def test_conservative_query(self):
        device = torch.device('cpu')
        data = load_linear_functions(device=device, **LOAD_ARGS)
        current_parameter = torch.zeros((LOAD_ARGS['num_params']+1,), dtype=LOAD_ARGS['dtype'], device=device)

        conservative_drift_query(current_parameter=current_parameter, planes=data)

    def test_conserative_query_cuda(self):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            data = load_linear_functions(device=device, **LOAD_ARGS)
            current_parameter = torch.zeros((LOAD_ARGS['num_params']+1,), dtype=LOAD_ARGS['dtype'], device=device)

            conservative_drift_query(current_parameter=current_parameter, planes=data)

if __name__ == '__main__':
    unittest.main()