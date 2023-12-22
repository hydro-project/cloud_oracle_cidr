import math

def compute_combinations(n: int, k: int) -> int:
    """
    Computes the number of combinations of choosing k items from a set of n items.

    Parameters:
        n (int): The total number of items in the set.
        k (int): The number of items to choose.

    Returns:
        int: The number of combinations.

    Example:
        >>> compute_combinations(5, 2)
        10
    """
    return int(math.factorial(n) / (math.factorial(k) * math.factorial(n - k)))