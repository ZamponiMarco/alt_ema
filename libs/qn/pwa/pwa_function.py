from typing import Tuple, Optional, Dict, Any, List

import cvxpy as cp
import numpy as np
from gurobipy import GRB
from pycvxset import Polytope
import gurobipy as gp


class Region:
    def __init__(self, polytope: Polytope, A: np.ndarray, b: float):
        self.polytope = polytope
        self.A = A  # Coefficients of the affine function
        self.b = b  # Offset of the affine function

    def contains(self, x: np.ndarray) -> bool:
        """Check if point x is inside the polytope."""
        return self.polytope.contains([x, x])[0]

    def get_h_repr(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the H-representation (A, b) of the polytope."""
        return self.polytope.A, self.polytope.b

    def evaluate(self, x: np.ndarray) -> float:
        """Evaluate the affine function at point x."""
        return float(np.dot(self.A, x) + self.b)

    def plot(self, ax: Any, fixed_vars: Optional[Dict[int, float]] = None, **kwargs) -> None:
        """Plot the polytope of the region."""
        if fixed_vars:
            sliced_polytope = self.polytope.slice(fixed_vars)
            if sliced_polytope:
                sliced_polytope.plot(ax=ax, **kwargs)
        else:
            self.polytope.plot(ax=ax, **kwargs)


class PWAFunction:
    def __init__(self, regions: List[Region], value_bounds: Tuple[float, float] = (-float("inf"), float("inf"))):
        """
        Initialize with regions and optional bounds for the function values.
        """
        self.regions = regions
        self.min_value, self.max_value = value_bounds
        if self.min_value < self.max_value:
            self.augment_with_clipping()

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate the function at x and clip to bounds.
        """
        for region in self.regions:
            if region.contains(x):
                return region.evaluate(x)
        raise ValueError("Input x is not in any defined region.")

    def augment_with_clipping(self):
        """
        Augment the current PWA function by splitting each region into up to three subregions:
          - One where f(x) < lower (evaluates to lower)
          - One where lower <= f(x) <= upper (evaluates as usual)
          - One where f(x) > upper (evaluates to upper)
        Returns a new PWAFunction with these subregions.
        """
        new_regions = []
        for region in self.regions:
            A, b = region.A, region.b

            poly_low = region.polytope.intersection_with_halfspaces(A, - b + self.min_value)
            if not poly_low.is_empty:
                new_regions.append(Region(poly_low, np.zeros(poly_low.dim), self.min_value))

            poly_mid = region.polytope.intersection_with_halfspaces(
                np.vstack([A, -A]), np.hstack([- b + self.max_value, b - self.min_value]))
            if not poly_mid.is_empty:
                new_regions.append(Region(poly_mid, A, b))

            poly_high = region.polytope.intersection_with_halfspaces(-A, b - self.max_value)
            if not poly_high.is_empty:
                new_regions.append(Region(poly_high, np.zeros(poly_high.dim), self.max_value))

        self.regions = new_regions


def get_pwa_model(controller: PWAFunction, data_vars, M):
    regions = len(controller.regions)

    z = cp.Variable(regions, boolean=True)  # Binary variable pointing at region
    eval = cp.Variable(regions)  # Value of the function

    constraints = [
        cp.sum(z) == 1  # Only one region is evaluated
    ]

    for region_idx in range(regions):
        region: Region = controller.regions[region_idx]
        A, b = region.get_h_repr()
        for row in range(A.shape[0]):
            constraints += [
                # station_data \in dom \implies z[region_idx] = true
                A[row] @ data_vars <= b[row] + M * (1 - z[region_idx])
            ]
        constraints += [
            # eval = z[region] * Ax + b
            eval[region_idx] <= region.A @ data_vars + region.b + M * (1 - z[region_idx]),
            eval[region_idx] >= region.A @ data_vars + region.b - M * (1 - z[region_idx]),
            eval[region_idx] <= M * z[region_idx],
            eval[region_idx] >= -M * z[region_idx]
        ]

    return constraints, cp.sum(eval)


def gurobi_get_pwa_model(model: gp.Model, controller, data_vars):
    """
    Build a piecewise affine model in gurobipy using operator overloading.

    Parameters:
      model      : A gurobipy Model instance.
      controller : A PWAFunction instance, whose .regions is a list of Region objects.
      data_vars  : A list (or similar indexable collection) of gurobipy variables.
      M          : Big-M constant.

    Returns:
      (constrs, value_expr) where:
         constrs    : A list of constraints added to the model.
         value_expr : A gurobipy LinExpr representing the evaluated value.
    """
    regions = len(controller.regions)
    
    z = model.addVars(regions, vtype=GRB.BINARY)
    value = model.addVar()

    model.addConstr(gp.quicksum(z) == 1)

    for region_idx in range(regions):
        region: Region = controller.regions[region_idx]
        A, b = region.get_h_repr()
        model.addConstrs((z[region_idx] == 1) >> (A[row] @ data_vars - b[row] <= 0) for row in range(A.shape[0]))
        model.addConstr((z[region_idx] == 1) >> (value == region.A @ data_vars + region.b))

    return value