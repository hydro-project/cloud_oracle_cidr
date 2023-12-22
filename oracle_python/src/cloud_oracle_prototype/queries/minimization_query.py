import torch

def init_query(*, planes: torch.Tensor, batch_size: int, num_functions: int, num_params: int, device: torch.device, dtype: torch.dtype):
    # Verify that planes has the correct shape
    assert planes.shape == (num_functions, num_params), f"Planes has shape {planes.shape} but should have shape {(num_functions, num_params)}"
    
    outputs = torch.zeros((batch_size, 1), device=device, dtype=dtype)
    outputs_index = torch.zeros((batch_size, 1), device=device, dtype=torch.int64)
    
    # Initialize inputs on the device
    #workload_shape = (batch_size, num_params, 1)
    #inputs = torch.empty(workload_shape,device=device, dtype=dtype)
    # Allocate inputs without additional dimension. This is deferred to the query.
    inputs = torch.empty((batch_size, num_params), device=device, dtype=dtype)
    
    #buf = torch.empty((batch_size, num_functions, 1), device=device, dtype=dtype)

    return (outputs, outputs_index, inputs)

def load_random_inputs(inputs: torch.Tensor):
        return inputs.exponential_()

@torch.compile(options={"trace.graph_diagram": False, "trace.enabled": False}, fullgraph=False, dynamic=False)
def minimization_query(*, planes: torch.Tensor, inputs: torch.Tensor, outputs: torch.Tensor, outputs_index: torch.Tensor):

    # Compute cost for each parameter and sum them and take the minimum
    #simulated_outputs[i:batch_size] = torch.matmul(functions, random_inputs).min()

    #torch.matmul(planes, inputs, out=buf)
    #torch.min(buf, dim=1, keepdim=False, out=(simulated_outputs, simulated_outputs_index))
    # tensor = tensor.reshape(1, tensor.shape[0], tensor.shape[1])
    planes_batch = planes.view(1, planes.shape[0], planes.shape[1])
    inputs_batch = inputs.view(inputs.shape[0], inputs.shape[1], 1)
    intermediate = torch.matmul(planes_batch, inputs_batch)
    torch.min(intermediate, dim=1, keepdim=False, out=(outputs, outputs_index))

    return (outputs, outputs_index)