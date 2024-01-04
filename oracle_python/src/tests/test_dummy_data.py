import unittest
import torch
from cloud_oracle_prototype.dummy_data import load_linear_functions, load_linear_function_as_batch_tensor

SHARED_ARGS = dict(
    num_functions = 200,
    num_params = 10,
    coeff_min = 0,
    coeff_max = 1,
    dtype = torch.float16,
    device = torch.device('cpu')
)

class TestDummyData(unittest.TestCase):


    def test_load_linear_functions(self):
        # Define your test parameters
        as_planes = False
        random = False

        # Call the function with the test parameters
        result = load_linear_functions(as_planes=as_planes, random=random, **SHARED_ARGS)

        # Check the shape of the result
        num_functions = SHARED_ARGS['num_functions']
        num_params = SHARED_ARGS['num_params']
        self.assertEqual(result.shape, (num_functions, num_params))

        # Check if the coefficients sum to 1
        device = SHARED_ARGS['device']
        dtype = SHARED_ARGS['dtype']
        self.assertTrue(torch.allclose(result.sum(dim=1), torch.ones(num_functions, dtype=dtype, device=device)))

    def test_load_linear_functions_as_planes(self):
        # Define your test parameters
        as_planes = True
        random = False

        # Call the function with the test parameters
        result = load_linear_functions(as_planes=as_planes, random=random, **SHARED_ARGS)

        # Check the shape of the result
        num_functions = SHARED_ARGS['num_functions']
        num_params = SHARED_ARGS['num_params']
        self.assertEqual(result.shape, (num_functions, num_params+1))

    def test_load_linear_functions_as_planes_random(self):
        # Define your test parameters
        random = True
        as_planes = True

        # Call the function with the test parameters
        result = load_linear_functions(as_planes=as_planes, random=random, **SHARED_ARGS)

        # Check the shape of the result
        num_functions = SHARED_ARGS['num_functions']
        num_params = SHARED_ARGS['num_params']
        self.assertEqual(result.shape, (num_functions, num_params+1))

        # Check if the coefficient are random
        device = SHARED_ARGS['device']
        dtype = SHARED_ARGS['dtype']
        self.assertFalse(torch.allclose(result.sum(dim=1), torch.ones(num_functions, dtype=dtype, device=device)))

    def test_load_linear_functions_as_planes_random_cuda(self):
        # Test if the function works on cuda, if cuda is available
        if torch.cuda.is_available():
            # Define your test parameters
            num_functions = 10
            num_params = 5
            coeff_min = 0
            coeff_max = 1
            dtype = torch.float32
            device = torch.device('cuda:0')
            as_planes = True

            # Call the function with the test parameters
            result = load_linear_functions(num_functions=num_functions, num_params=num_params, coeff_min=coeff_min, coeff_max=coeff_max, dtype=dtype, device=device, as_planes=as_planes)

            # Check the shape of the result
            self.assertEqual(result.shape, (num_functions, num_params+1))

    def test_load_linear_function_as_batch_tensor(self):
        # Define your test parameters
        num_functions = 10
        num_params = 5
        coeff_min = 0
        coeff_max = 1
        dtype = torch.float32
        device = torch.device('cpu')

        # Call the function with the test parameters
        result = load_linear_function_as_batch_tensor(num_functions=num_functions, num_params=num_params, coeff_min=coeff_min, coeff_max=coeff_max, dtype=dtype, device=device)

        # Check the shape of the result
        self.assertEqual(result.shape, (1, num_functions, num_params))

if __name__ == '__main__':
    unittest.main()