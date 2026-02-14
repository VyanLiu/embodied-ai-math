import random
import math

from typing import List, Tuple, Callable, Optional

#Metric Spaces
#2.15 Definition A set X whose elements we shall call points, is said to be a metric space,
#if with any two points p and q of X, there's associated a real number d(p, q), called the distance from p to q,
#such that
#     (a) d(p, q) > 0 if p != q; d(p, p) = 0
#     (b) d(p, q) = d(q, p);    #symmetric
#     (c) d(p, q) <= d(p, r) + d(r, q), for any r belonging to X
#   Any function with these three properties is called a distance function, or a metric

#2.18 Definition Let X be a metric space. All points and sets mentioned below are understood to be elements and subsets of X.
#     (a) A neighborhood of p is a set Nr(p) consisting of all q such that d(p, q) < r, for some r > 0. The number r is called
#         the radius of Nr(p).
#     (b) A point p is a limit point of the set E if every neighborhood of p contains a point q != p such that q belongs to E.
#     (c) If p belongs to E and p is not a limit point of E, then p is called an isolated point of E.
#     (d) E is closed if every limit point of E is a point of E.
#     (e) A point p is an interior point of E if there is a neighborhood N of p such that N is included by E.
#     (f) E is open if every point of E is an interior point of E.
#     (g) The complement of E (denoted by E^c) is the set of all points p belonging to X such that p doesn't belong to E.
#     (h) E is perfect if E is closed and if every point of E is a limit point of E
#     (i) E is bounded if there's a real number M and a point q belonging to X such that d(p, q) < M for all p belong to E.
#     (j) E is dense in X if every point of X is a limit point of E, or a point of E or both.

#Define a class type of Metric spaces, use different type of Distance to achieve verify different spaces as metric spaces
def are_points_equivalent(p1: Tuple[float, ...], p2: Tuple[float,...], eps: float = 1e-9) -> bool:
    """
    Check if two points are equivalent within a given tolerance.
    
    Args:
        p1: First point as a tuple of floats
        p2: Second point as a tuple of floats
        eps: Tolerance level for comparison (default 1e-9)
        
    Returns:
        True if points are equivalent within tolerance, False otherwise
    """
    if len(p1) != len(p2):
        return False
    return all(abs(a - b) <= eps for a, b in zip(p1, p2))


class MetricSpace:
    """
    A class representing a metric space.
    
    A metric space is a set X whose elements we shall call points, 
    with an associated distance function d that satisfies three axioms:
    (a) d(p, q) > 0 if p != q; d(p, p) = 0
    (b) d(p, q) = d(q, p) (symmetry)
    (c) d(p, q) <= d(p, r) + d(r, q) (triangle inequality)
    """
    
    def __init__(self, x: List[Tuple[float, ...]], d: Callable[[Tuple[float, ...], Tuple[float, ...]], float]):
        """
        Initialize a metric space.
        
        Args:
            x: List of points in the space, each point is a tuple of floats
            d: Distance function that takes two points and returns a float
        """
        if not x:
            raise ValueError("Metric space cannot be initialized with an empty set of points")
        self.x = x
        self.d = d
        self.number = len(self.x)
        self.tuple_dimension = len(self.x[0])

    def verify_axioms(self, num_tests: int = 100) -> bool:
        """
        Verify the three axioms of metric spaces through random sampling.
        
        Args:
            num_tests: Number of random tests to perform (default 100)
            
        Returns:
            True if all axioms are satisfied in the tests, False otherwise
        """
        if num_tests <= 0:
            raise ValueError("Number of tests must be positive")
        
        all_passed = True
        for _ in range(num_tests):
            p, q, r = random.choices(self.x, k=3)

            try:
                dist_pq = self.d(p, q)
                dist_qp = self.d(q, p)
                dist_pp = self.d(p, p)
                dist_pr = self.d(p, r)
                dist_rq = self.d(r, q)
            except Exception as e:
                print(f"Error computing distance: {e}")
                all_passed = False
                continue

            # Validate that distances are non-negative numbers
            if not isinstance(dist_pq, (int, float)) or dist_pq < 0:
                print(f"Distance function returned invalid value: {dist_pq}")
                all_passed = False
                continue

            # (a) positive property: d(p, q) > 0 if p != q; d(p, p) = 0
            if p != q and dist_pq <= 0:
                print(f"Axiom (a) failed: d({p}, {q}) = {dist_pq} <= 0 but p != q")
                all_passed = False
            if dist_pp != 0:
                print(f"Axiom (a) failed: d({p}, {p}) = {dist_pp} != 0")
                all_passed = False
            # (b) symmetric property: d(p, q) = d(q, p)
            if not math.isclose(dist_pq, dist_qp):
                print(f"Axiom (b) failed: d({p}, {q}) = {dist_pq} != d({q}, {p}) = {dist_qp}")
                all_passed = False
            # (c) triangle inequality: d(p, q) <= d(p, r) + d(r, q)
            if dist_pq > dist_pr + dist_rq:
                print(f"Axiom (c) failed: d({p}, {q}) = {dist_pq} > d({p}, {r}) + d({r}, {q}) = {dist_pr + dist_rq}")
                all_passed = False

        if all_passed:
            print("All tests passed, this space is a metric space")
        else:
            print("This space is not a metric space")

        return all_passed

    def find_neighborhood(self, p: Tuple[float, ...], r: float) -> List[Tuple[float, ...]]:
        """
        Find the neighborhood of a point p with radius r.
        
        A neighborhood of p is a set Nr(p) consisting of all q such that d(p, q) < r.
        
        Args:
            p: Center point of the neighborhood
            r: Radius of the neighborhood
            
        Returns:
            List of points in the neighborhood
        """
        if r < 0:
            raise ValueError("Radius must be non-negative")
        
        nr: List[Tuple[float, ...]] = []
        for point in self.x:
            try:
                distance = self.d(p, point)
                if not isinstance(distance, (int, float)):
                    raise TypeError(f"Distance function must return a number, got {type(distance)}")
                if distance < r:
                    nr.append(point)
            except Exception as e:
                print(f"Error computing distance between {p} and {point}: {e}")
                continue
        return nr

    def is_limit_point(self, p: Tuple[float, ...], e: List[Tuple[float, ...]], epsilon_list: Optional[List[float]] = None) -> bool:
        """
        Check if a point p is a limit point of set E.
        
        A point p is a limit point of the set E if every neighborhood of p 
        contains a point q != p such that q belongs to E.
        
        Args:
            p: Point to check if it's a limit point
            e: Set E to check against
            epsilon_list: List of radii to test neighborhoods (defaults to [10^-k for k in range(1, 11)])
            
        Returns:
            True if p is a limit point of E, False otherwise
        """
        if epsilon_list is not None:
            if not all(isinstance(eps, (int, float)) and eps > 0 for eps in epsilon_list):
                raise ValueError("All epsilon values must be positive numbers")
        
        # Since computers cannot check all epsilon > 0, we approximate this by testing a sequence of sufficiently small epsilon values
        # Set default epsilon list value
        f_is_limit_point = False
        if epsilon_list is None:
            epsilon_list = [10**-k for k in range(1, 11)]

        for eps in epsilon_list:
            try:
                # Use enough neighborhoods instead of arbitrary ones
                nr = self.find_neighborhood(p, eps)
                for q in e:
                    # Test if q is one of elements in every neighborhood of p and q is not equal to p
                    if q in nr and not are_points_equivalent(q, p):
                        f_is_limit_point = True
                        break
                if f_is_limit_point:
                    break
            except Exception as ex:
                print(f"Error processing neighborhood with radius {eps}: {ex}")
                continue
                
        if f_is_limit_point:
            print("p is a limit point of the set E")
        else:
            print("p is not a limit point of the set E")
        return f_is_limit_point

    def is_isolated_point(self, p: Tuple[float, ...], e: List[Tuple[float, ...]]) -> bool:
        """
        Check if a point p is an isolated point of set E.
        
        A point p is an isolated point of E if p belongs to E and p is not a limit point of E.
        
        Args:
            p: Point to check if it's an isolated point
            e: Set E to check against
            
        Returns:
            True if p is an isolated point of E, False otherwise
        """
        if p not in e or self.is_limit_point(p, e):
            f_is_isolated_point = False
        else:
            f_is_isolated_point = True

        return f_is_isolated_point

    def is_interior_point(self, p: Tuple[float, ...], e: List[Tuple[float, ...]], r: float) -> bool:
        """
        Check if a point p is an interior point of set E.
        
        A point p is an interior point of E if there is a neighborhood N of p such that N is included in E.
        
        Args:
            p: Point to check if it's an interior point
            e: Set E to check against
            r: Radius of the neighborhood to test
            
        Returns:
            True if p is an interior point of E, False otherwise
        """
        if r < 0:
            raise ValueError("Radius must be non-negative")
        
        try:
            nr = self.find_neighborhood(p, r)
        except Exception as ex:
            print(f"Error finding neighborhood: {ex}")
            return False
        
        # Check if every point in the neighborhood is in E
        for q in nr:
            # If any point in the neighborhood is not in E, p is not an interior point
            if not any(are_points_equivalent(q, ele) for ele in e):
                print("p is not an interior point of the set E")
                return False
        
        print("p is an interior point of the set E")
        return True

    def is_closed_set(self, e: List[Tuple[float, ...]]) -> bool:
        """
        Check if a set E is closed.
        
        A set E is closed if every limit point of E is a point of E.
        
        Args:
            e: Set E to check if it's closed
            
        Returns:
            True if E is closed, False otherwise
        """
        # If e is empty, it's closed by definition
        if not e:
            return True
        
        # Check if any point in the metric space \ E is a limit point of E
        points_not_in_e = [point for point in self.x if not any(are_points_equivalent(point, p) for p in e)]
        for point in points_not_in_e:
            if self.is_limit_point(point, e):
                return False  # Found a limit point of E that's not in E
        return True

    def is_open_set(self, e: List[Tuple[float, ...]], r: float) -> bool:
        """
        Check if a set E is open.
        
        A set E is open if every point of E is an interior point of E.
        
        Args:
            e: Set E to check if it's open
            r: Radius to use for checking interior points
            
        Returns:
            True if E is open, False otherwise
        """
        if r < 0:
            raise ValueError("Radius must be non-negative")
        
        f_is_open_set = True
        for p in e:
            try:
                if not self.is_interior_point(p, e, r):
                    f_is_open_set = False
                    break
            except Exception as ex:
                print(f"Error checking if {p} is an interior point: {ex}")
                f_is_open_set = False
                break
        return f_is_open_set

    def find_complement_set(self, e: List[Tuple[float, ...]]) -> List[Tuple[float, ...]]:
        """
        Find the complement of set E in the metric space.
        
        The complement of E (denoted by E^c) is the set of all points p 
        belonging to X such that p doesn't belong to E.
        
        Args:
            e: Set E to find the complement of
            
        Returns:
            Complement set E^c
        """
        if e is None:
            raise ValueError("Set E cannot be None")
        
        complement_set: List[Tuple[float, ...]] = []
        for p in self.x:
            try:
                f_find_in_set = any(are_points_equivalent(p, q) for q in e)
                if not f_find_in_set:
                    complement_set.append(p)
            except Exception as ex:
                print(f"Error checking if {p} is in set E: {ex}")
                continue

        return complement_set

    def is_perfect(self, e: List[Tuple[float, ...]]) -> bool:
        """
        Check if a set E is perfect.
        
        A set E is perfect if E is closed and if every point of E is a limit point of E.
        
        Args:
            e: Set E to check if it's perfect
            
        Returns:
            True if E is perfect, False otherwise
        """
        if e is None:
            raise ValueError("Set E cannot be None")
        
        # Check if every point in E is a limit point of E
        f_limit_sets_verified = True
        for p in e:
            try:
                if not self.is_limit_point(p, e):
                    f_limit_sets_verified = False
                    break
            except Exception as ex:
                print(f"Error checking if {p} is a limit point of E: {ex}")
                f_limit_sets_verified = False
                break

        # Check if E is closed
        try:
            is_closed = self.is_closed_set(e)
        except Exception as ex:
            print(f"Error checking if E is closed: {ex}")
            return False
        
        if is_closed and f_limit_sets_verified:
            print("The set is perfect")
            f_is_perfect = True
        else:
            print("The set is not a perfect set")
            f_is_perfect = False

        return f_is_perfect