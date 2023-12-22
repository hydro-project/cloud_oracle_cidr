import unittest
import torch
from oracle.dummy_data import load_planes_with_savings
from oracle.queries.simulation_query import init_query, simulation_query

LOAD_ARGS = dict(
    num_functions = 100,
    num_functions_other = 100,
    num_params = 10,
    coeff_min = 0,
    coeff_max = 1,
    dtype = torch.float16,
    as_planes = True,
    random = False
)

class TestSimulationQuery(unittest.TestCase):

    def test_simulation_query():

        sample_size = 10**6
        batch_size = sample_size
        device = torch.device('cpu')

        # Load planes with simulated savings
        # Simulate savings for new planes: Savings for EU data centers, slight increase for US, and unchanged for other data centers
        percent_us_data_centers = 0.3
        percent_eu_data_centers = 0.3
        percent_change_us = 1.1
        percent_change_eu = 0.1

        planes, planes_other = load_planes_with_savings(num_functions=LOAD_ARGS['num_functions'], num_functions_other=LOAD_ARGS['num_functions_other'], num_params=LOAD_ARGS['num_params'], device=device, dtype=LOAD_ARGS['dtype'], percent_data_centers_A=percent_us_data_centers, percent_data_centers_B=percent_eu_data_centers, percent_change_A=percent_change_us, percent_change_B=percent_change_eu)
        
        # Initialize query state
        (outputs, inputs)= init_query(planes=planes, planes_other=planes_other, batch_size=batch_size, num_params=LOAD_ARGS['num_params'], device=device, dtype=LOAD_ARGS['dtype'])

        #inputs.exponential_()
        inputs.uniform_()
        #inputs.fill_(1.0)
        
        # Run query
        res = simulation_query(planes=planes, planes_other=planes_other, inputs=inputs, outputs=outputs)
        #print(res)

        # Get median savings and the pth confidence interval
        median_savings = res[0].item()
        confidence_interval = (res[1].item(), res[2].item())
        print(f"Sample size: {sample_size}, median savings: {median_savings}, confidence: {confidence_interval}")

    def test_simulation_query_cuda():
        if torch.cuda.is_available():
            sample_size = 10**6
            batch_size = sample_size
            device = torch.device('cuda:0')

            # Load planes with simulated savings
            # Simulate savings for new planes: Savings for EU data centers, slight increase for US, and unchanged for other data centers
            percent_us_data_centers = 0.3
            percent_eu_data_centers = 0.3
            percent_change_us = 1.1
            percent_change_eu = 0.1

            planes, planes_other = load_planes_with_savings(num_functions=LOAD_ARGS['num_functions'], num_functions_other=LOAD_ARGS['num_functions_other'], num_params=LOAD_ARGS['num_params'], device=device, dtype=LOAD_ARGS['dtype'], percent_data_centers_A=percent_us_data_centers, percent_data_centers_B=percent_eu_data_centers, percent_change_A=percent_change_us, percent_change_B=percent_change_eu)
            
            # Initialize query state
            (outputs, inputs)= init_query(planes=planes, planes_other=planes_other, batch_size=batch_size, num_params=LOAD_ARGS['num_params'], device=device, dtype=LOAD_ARGS['dtype'])

            #inputs.exponential_()
            inputs.uniform_()
            #inputs.fill_(1.0)
            
            # Run query
            res = simulation_query(planes=planes, planes_other=planes_other, inputs=inputs, outputs=outputs)
            #print(res)

            # Get median savings and the pth confidence interval
            median_savings = res[0].item()
            confidence_interval = (res[1].item(), res[2].item())
            print(f"Sample size: {sample_size}, median savings: {median_savings}, confidence: {confidence_interval}")
        else:
            print("CUDA not available")

if __name__ == '__main__':
    unittest.main()