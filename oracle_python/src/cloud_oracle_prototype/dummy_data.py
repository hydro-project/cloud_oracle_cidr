import torch

def load_linear_functions(*, num_functions, num_pieces = 1, num_params, coeff_min = 0, coeff_max = 1, dtype, device, as_planes=False, random=False):
    
    # Add one parameter for the additional dimension of planes
    if as_planes:
        num_params += 1

    # Shape of the tensor: (num_functions, num_params)
    shape = (num_functions, num_params)

    functions_tensor = torch.empty(shape, dtype=dtype, device=device)

    if random:
        # Fill with random values
        functions_tensor.uniform_(coeff_min, coeff_max)
    else:
        # Fill with coefficients summing to 1
        functions_tensor.fill_(1/num_params)

    return functions_tensor

def load_linear_function_as_batch_tensor(**kwargs):
    tensor = load_linear_functions(**kwargs)
    # Reshape tensor for batch, to (1, num_functions, num_pieces)
    tensor = tensor.reshape(1, tensor.shape[0], tensor.shape[1])
    return tensor

def load_planes_with_savings(*, num_functions: int, num_functions_other: int, num_params: int, device: str, dtype, percent_data_centers_A: float, percent_data_centers_B: float, percent_change_A: float, percent_change_B: float) -> (torch.Tensor, torch.Tensor):

    # Load planes and other planes
    planes = load_linear_functions(num_functions=num_functions, num_params=num_params, dtype=dtype, device=device, as_planes=True, random=False)
    planes_other = load_linear_functions(num_functions=num_functions_other, num_params=num_params, dtype=dtype, device=device, as_planes=True, random=False)

    # Shift planes
    end_a = int(num_params * percent_data_centers_A)
    end_b = int(end_a + num_params * percent_data_centers_B)
    planes[:, 0:end_a] *= percent_change_A
    planes[:, end_a:end_b] *= percent_change_B

    return (planes, planes_other)