#Chapter 2 Code Achievement
#Basic Topology
"""
This module implements basic topological concepts based on Chapter 2 of Rudin's Principles of Mathematical Analysis.
It includes various distance functions and examples of metric spaces.
"""
import math
from typing import Tuple

import metric_space
from math_notes.rudin_math.convergence.metric_space import MetricSpace

#Euclidean Distance
def cal_euclidean_distance(p: Tuple[float, ...], q: Tuple[float, ...]) -> float:
    """
    Calculate the Euclidean distance between two points.
    
    Args:
        p: First point as a tuple of coordinates
        q: Second point as a tuple of coordinates
        
    Returns:
        Euclidean distance between p and q
        
    Raises:
        ValueError: If points have different dimensions
    """
    if len(p) != len(q):
        raise ValueError(f"Points must have the same dimension. Got {len(p)} and {len(q)}")
    diffs = [(pi - qi)**2 for pi, qi in zip(p, q)]
    return math.hypot(*diffs)

def cal_manhattan_distance(p: Tuple[float, ...], q: Tuple[float, ...]) -> float:
    """
    Calculate the Manhattan (taxicab) distance between two points.
    
    Args:
        p: First point as a tuple of coordinates
        q: Second point as a tuple of coordinates
        
    Returns:
        Manhattan distance between p and q
        
    Raises:
        ValueError: If points have different dimensions
    """
    if len(p) != len(q):
        raise ValueError(f"Points must have the same dimension. Got {len(p)} and {len(q)}")
    
    return sum(math.fabs(pi - qi) for pi, qi in zip(p, q))

def cal_chebyshev_distance(p: Tuple[float, ...], q: Tuple[float, ...]) -> float:
    """
    Calculate the Chebyshev (maximum) distance between two points.
    
    Args:
        p: First point as a tuple of coordinates
        q: Second point as a tuple of coordinates
        
    Returns:
        Chebyshev distance between p and q
        
    Raises:
        ValueError: If points have different dimensions
    """
    if len(p) != len(q):
        raise ValueError(f"Points must have the same dimension. Got {len(p)} and {len(q)}")
    
    return max(math.fabs(pi - qi) for pi, qi in zip(p, q))

def cal_minkowski_distance(p: Tuple[float, ...], q: Tuple[float, ...], p_order: float = 2) -> float:
    """
    Calculate the Minkowski distance between two points.
    
    Args:
        p: First point as a tuple of coordinates
        q: Second point as a tuple of coordinates
        p_order: Order of the Minkowski distance (p=1 gives Manhattan, p=2 gives Euclidean)
        
    Returns:
        Minkowski distance between p and q
        
    Raises:
        ValueError: If points have different dimensions or p_order is not positive
    """
    if len(p) != len(q):
        raise ValueError(f"Points must have the same dimension. Got {len(p)} and {len(q)}")
    if p_order <= 0:
        raise ValueError(f"Order p must be positive. Got {p_order}")
    
    return sum(abs(pi - qi)**p_order for pi, qi in zip(p, q))**(1/p_order)

def cal_discrete_metric(p: Tuple[float, ...], q: Tuple[float, ...], tolerance: float = 1e-9) -> float:
    """
    Calculate the discrete metric between two points.
    
    Args:
        p: First point as a tuple of coordinates
        q: Second point as a tuple of coordinates
        tolerance: Tolerance for comparing equality of points
        
    Returns:
        0 if points are equal, 1 otherwise
        
    Raises:
        ValueError: If points have different dimensions
    """
    if len(p) != len(q):
        raise ValueError(f"Points must have the same dimension. Got {len(p)} and {len(q)}")
    
    # Check if points are equivalent within tolerance
    if all(abs(pi - qi) < tolerance for pi, qi in zip(p, q)):
        return 0
    else:
        return 1

def cal_fake_metric_distance(p: Tuple[float, ...], q: Tuple[float, ...]) -> float:
    """
    A function that does NOT satisfy all metric space axioms (not a true metric).
    This is used to demonstrate what happens when the axioms are violated.
    
    Args:
        p: First point as a tuple of coordinates
        q: Second point as a tuple of coordinates
        
    Returns:
        A distance-like value that doesn't satisfy all metric axioms
        
    Raises:
        ValueError: If points have different dimensions
    """
    if len(p) != len(q):
        raise ValueError(f"Points must have the same dimension. Got {len(p)} and {len(q)}")
    
    # This only considers the first coordinate, violating the identity of indiscernibles
    return math.fabs(p[0] - q[0])

if __name__ == '__main__':
    # Define sample points in 3D space
    A = [
        (3.499999999999, 2.40000000000001, 4.7999999999999), 
        (1.2, 3.7, 0.8), 
        (0.9, 0.4, 1.8), 
        (0.2, 9.1, 100.2), 
        (2.6, 7.4, 1.4),
        (8.5, 1.2, 0.9), 
        (1.2, 0.2, 4.5), 
        (3.2, 18.23, 0.99)
    ]
    
    # Define a subset E of A
    E = [
        (3.499999999999, 2.40000000000001, 4.7999999999999),
        (0.9, 0.4, 1.8), 
        (3.7, 2.1, 1.25),  # Fixed syntax error: was (3.7, 2.1, 1,25)
        (1.2, 4.5, 3.8)    # Fixed syntax error: was (1.2, 4.5, 3,8)
    ]
    
    p1 = (3.5, 2.4, 4.8)
    p2 = (0.89, 0.41, 1.799)
    
    print("=== Basic Topology Examples ===")
    print(f"Sample space A: {A}")
    print(f"Subset E: {E}")
    print(f"Point p1: {p1}")
    print(f"Point p2: {p2}")
    
    print("\n1. Verify Axioms for Euclidean space:")
    try:
        euclidean_space = MetricSpace(A, cal_euclidean_distance)
        euclidean_space.verify_axioms()
        euclidean_space.is_limit_point(p1, E)
    except Exception as e:
        print(f"Error with Euclidean space: {e}")

    print("\n2. Verify Axioms for Manhattan space:")
    try:
        manhattan_space = MetricSpace(A, cal_manhattan_distance)
        manhattan_space.verify_axioms()
    except Exception as e:
        print(f"Error with Manhattan space: {e}")

    print("\n3. Verify Axioms for Chebyshev space:")
    try:
        chebyshev_space = MetricSpace(A, cal_chebyshev_distance)
        chebyshev_space.verify_axioms()
    except Exception as e:
        print(f"Error with Chebyshev space: {e}")

    print("\n4. Verify Axioms for Minkowski space (p=3):")
    try:
        minkowski_space = MetricSpace(A, lambda p, q: cal_minkowski_distance(p, q, 3))
        minkowski_space.verify_axioms()
    except Exception as e:
        print(f"Error with Minkowski space: {e}")

    print("\n5. Verify Axioms for Discrete metric space:")
    try:
        discrete_space = MetricSpace(A, cal_discrete_metric)
        discrete_space.verify_axioms()
    except Exception as e:
        print(f"Error with Discrete metric space: {e}")

    print("\n6. Verify Axioms for fake metric space (should fail):")
    try:
        fake_metric_space = MetricSpace(A, cal_fake_metric_distance)
        fake_metric_space.verify_axioms()
    except Exception as e:
        print(f"Error with fake metric space: {e}")

    print("\n7. Testing basic topology concepts:")
    try:
        # Create a space with Euclidean distance
        space = MetricSpace(A, cal_euclidean_distance)
        
        # Test if p1 is a limit point of E
        print(f"\nIs {p1} a limit point of E? {space.is_limit_point(p1, E)}")
        
        # Test if p2 is a limit point of E
        print(f"Is {p2} a limit point of E? {space.is_limit_point(p2, E)}")
        
        # Test if a point from E is isolated
        if E:
            print(f"Is {E[0]} an isolated point of E? {space.is_isolated_point(E[0], E)}")
        
        # Test if E is closed
        print(f"Is E closed? {space.is_closed_set(E)}")
        
        # Test if E is open (using a small radius)
        print(f"Is E open? {space.is_open_set(E, 0.1)}")
        
        # Find complement of E in A
        complement_E = space.find_complement_set(E)
        print(f"Complement of E in A: {complement_E}")
        
        # Test if E is perfect
        print(f"Is E perfect? {space.is_perfect(E)}")
        
    except Exception as e:
        print(f"Error testing topology concepts: {e}")