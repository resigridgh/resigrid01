import secrets
import math
from typing import Optional, List, Union
import numpy as np

def uniform(a: float = 0.0, b: float = 1.0) -> float:

        if b <= a:
        raise ValueError("b must be greater than a")

    # 53 random bits gives 53-bit precision double
    u = secrets.randbits(53) / (1 << 53)  # in [0, 1)
    return a + (b - a) * u

def exponentialdist(lambd: float = 1.0) -> float:


        if lambd <= 0:
        raise ValueError("lambda must be greater than 0")

    # Generate cryptographically secure uniform sample
    u = uniform(0.0, 1.0)

    # Apply inverse transform sampling
    return -math.log(u) / lambd

def poissondist(lambd: float = 1.0, max_iter: int = 1000) -> int:


    if lambd <= 0:
        raise ValueError("lambda must be greater than 0")

    # Generate cryptographically secure uniform sample
    u = uniform(0.0, 1.0)

    # Inverse transform sampling for Poisson distribution
    k = 0
    p = math.exp(-lambd)  # P(X = 0)
    F = p                  # F(0) = P(X <= 0)

    # Iterate until we find the smallest k such that F(k) >= u
    iteration = 0
    while u > F and iteration < max_iter:
        k += 1
        # Update probability using recurrence relation: P(X=k) = (λ/k) * P(X=k-1)
        p = p * lambd / k
        F += p
        iteration += 1

    if iteration == max_iter:
        raise RuntimeError("Maximum iterations reached in Poisson sampling")

    return k

def generate_samples(distribution_func, n: int, *args, **kwargs) -> List:

    return [distribution_func(*args, **kwargs) for _ in range(n)]

def discrete_inverse_transform(pmf_values: List[float], outcomes: List[int]) -> int:

    if abs(sum(pmf_values) - 1.0) > 1e-10:
        raise ValueError("PMF values must sum to 1")

    u = uniform(0.0, 1.0)
    F = 0.0

    for i, prob in enumerate(pmf_values):
        F += prob
        if u <= F:
            return outcomes[i]

    return outcomes[-1]
