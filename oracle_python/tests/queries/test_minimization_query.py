import unittest
import torch
from oracle.dummy_data import load_linear_functions
from oracle.queries.minimization_query import init_query, minimization_query

LOAD_ARGS = dict(
    num_functions = 200,
    num_params = 10,
    coeff_min = 0,
    coeff_max = 1,
    dtype = torch.float16,
    as_planes = False,
    random = False
)

class TestDirectedDriftQuery(unittest.TestCase):
    def test_minimization_query(self):
        batch_size = 1
        device = torch.device('cpu')
        data = load_linear_functions(device=device, **LOAD_ARGS)
        
        # Initialize query state
        (outputs, output_indexes, inputs)= init_query(planes=data, batch_size=batch_size, num_functions=LOAD_ARGS['num_functions'], num_params=LOAD_ARGS['num_params'], device=device, dtype=LOAD_ARGS['dtype'])

        inputs.zero_()
        
        # Run query
        minimization_query(planes=data, inputs=inputs, outputs=outputs, outputs_index=output_indexes)

    def test_minimization_query_batched(self):
        batch_size = 64
        device = torch.device('cpu')
        data = load_linear_functions(device=device, **LOAD_ARGS)
        
        # Initialize query state
        (outputs, output_indexes, inputs)= init_query(planes=data, batch_size=batch_size, num_functions=LOAD_ARGS['num_functions'], num_params=LOAD_ARGS['num_params'], device=device, dtype=LOAD_ARGS['dtype'])

        inputs.zero_()
        
        # Run query
        minimization_query(planes=data, inputs=inputs, outputs=outputs, outputs_index=output_indexes)

    def test_minimization_query_cuda(self):
        if torch.cuda.is_available():
            batch_size = 1
            device = torch.device('cuda:0')
            data = load_linear_functions(device=device, **LOAD_ARGS)
            
            # Initialize query state
            (outputs, output_indexes, inputs)= init_query(planes=data, batch_size=batch_size, num_functions=LOAD_ARGS['num_functions'], num_params=LOAD_ARGS['num_params'], device=device, dtype=LOAD_ARGS['dtype'])

            inputs.zero_()
            
            # Run query
            minimization_query(planes=data, inputs=inputs, outputs=outputs, outputs_index=output_indexes)
        else:
            print("CUDA not available")

    def test_minimization_query_cuda_batched(self):
        if torch.cuda.is_available():
            batch_size = 64
            device = torch.device('cuda:0')
            data = load_linear_functions(device=device, **LOAD_ARGS)
            
            # Initialize query state
            (outputs, output_indexes, inputs)= init_query(planes=data, batch_size=batch_size, num_functions=LOAD_ARGS['num_functions'], num_params=LOAD_ARGS['num_params'], device=device, dtype=LOAD_ARGS['dtype'])

            inputs.zero_()
            
            # Run query
            minimization_query(planes=data, inputs=inputs, outputs=outputs, outputs_index=output_indexes)
        else:
            print("CUDA not available")

if __name__ == '__main__':
    unittest.main()