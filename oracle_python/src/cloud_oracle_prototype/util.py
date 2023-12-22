import torch
import os

DTYPE = torch.float16
def load_device(device_name):
    # Check if CUDA is available, else use CPU
    if "cuda" in device_name and torch.cuda.is_available():
        device = torch.device(device_name)
    elif "mps" in device_name and torch.backends.mps.is_available():
        # Fallback to CPU based implementation of operators if MPS based implementation is not available
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1"
        #print(os.environ["PYTORCH_ENABLE_MPS_FALLBACK"])
        device = torch.device(device_name)
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Set default data type
    torch.set_default_dtype(DTYPE)

    return device

def check_device_type_available(device_name):
    if "cuda" in device_name and not torch.cuda.is_available():
        return False
    elif "mps" in device_name and not torch.backends.mps.is_available():
        return False
    else:
        return True