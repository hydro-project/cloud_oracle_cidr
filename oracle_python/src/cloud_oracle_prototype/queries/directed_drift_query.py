import torch

@torch.compile(options={"trace.graph_diagram": False, "trace.enabled": False}, fullgraph=False, dynamic=False)
def directed_drift_query(*, current_parameter: torch.Tensor, drift: torch.Tensor, planes: torch.Tensor):
    """
    Computes the distance and index of the next optimal decision given the current parameter,
    a drift, and the planes of all decisions in the oracle.

    Args:
        current_parameter (torch.Tensor): The current parameter.
        drift (torch.Tensor): The drift vector.
        planes (torch.Tensor): The planes matrix.

    Returns:
        distance (torch.Tensor): The distance to the next optimal decision.
        next_index (torch.Tensor): The index of the next optimal decision.
    """    

    current_parameter[-1] = 0

    # Get current minimum decision
    curr = torch.matmul(planes, current_parameter).min(dim=0)
    index  = curr.indices
    value  = curr[0]

    # Point on plane of current minimum decision
    current_parameter[-1] = value

    # Project drift onto current minimum decision
    projected_drift = drift - (torch.dot(drift, planes[index])) * planes[index]

    ## Mask out current minimum decision
    # Copy intercept of current minimum decision
    temp = planes[index][0]
    planes[index][0] = torch.finfo(planes.dtype).max

    # Ray shooting from current minimum point in direction of projected drift
    res = ((planes.matmul(current_parameter)) / (planes.matmul(projected_drift))).min(dim=0)
    distance = res[0]

    # Index of next optimal decision
    next_index = res.indices.squeeze(dim=-1)
    
    # Restore plane of current minimum function
    planes[index][0] = temp.item()

    return distance, next_index