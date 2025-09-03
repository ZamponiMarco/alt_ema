import numpy as np
from scipy.optimize import linprog
from scipy.linalg import null_space
from libs.qn.examples.closed_queuing_network import example1, acmeair_qn
from libs.qn.examples.controller import constant_controller
from libs.qn.model.queuing_network import ClosedQueuingNetwork
from libs.qn.examples.controller import autoscalers

def build_polytope(horizon, skewness, l_bounds=(0.0, 1.0), l0 = 10):
    """
    Build polytope constraints A, b for variables l[0..horizon-1].
    Polytope = {x | A x <= b}.
    """
    n = horizon
    k = int(0.15 * horizon)
    delta = skewness

    A = []
    b = []

    # Initial condition: l[0] == l0
    row = np.zeros(n); row[0] = 1
    A.append(row); b.append(l0)
    A.append(-row); b.append(-l0)

    # Warm-up equalities: l[t] - l[t-1] == 0
    for t in range(1, k):
        row = np.zeros(n); row[t] = 1; row[t-1] = -1
        A.append(row); b.append(0.0)
        A.append(-row); b.append(0.0)

    # Rest inequalities: -delta <= l[t]-l[t-1] <= delta
    for t in range(k, n):
        row = np.zeros(n); row[t] = 1; row[t-1] = -1
        A.append(row); b.append(delta)
        A.append(-row); b.append(delta)

    # Box constraints
    for t in range(n):
        row = np.zeros(n); row[t] = 1
        A.append(row); b.append(l_bounds[1])
        A.append(-row); b.append(-l_bounds[0])

    return np.array(A), np.array(b)


def is_feasible(x, A, b):
    return np.all(A.dot(x) <= b + 1e-9)


def hit_and_run(A, b, x0, n_samples, burn_in=1000, steps_per_sample=100):
    """
    Basic Hit-and-Run sampler for Ax <= b, with support for equality constraints.
    Projects directions onto the null space of equality constraints.
    """
    n = len(x0)
    x = x0.copy()
    samples = []

    # Identify equality constraints (pairs: row and -row with b and -b)
    eq_indices = []
    i = 0
    while i < len(b) - 1:
        if np.allclose(A[i], -A[i+1]) and np.allclose(b[i], -b[i+1]):
            eq_indices.append(i)
            i += 2  # Skip the paired inequality
        else:
            i += 1

    # Build equality matrix A_eq from identified pairs
    A_eq = A[eq_indices] if eq_indices else np.empty((0, n))
    if A_eq.shape[0] > 0:
        # Compute null space basis
        null_basis = null_space(A_eq)
    else:
        null_basis = np.eye(n)  # No equalities, use full space

    total_steps = burn_in + n_samples * steps_per_sample
    for step in range(total_steps):
        # Sample random direction in the null space
        coeffs = np.random.normal(size=null_basis.shape[1])
        d = null_basis.dot(coeffs)
        d /= np.linalg.norm(d) if np.linalg.norm(d) > 0 else 1

        Ad = A.dot(d)
        Ax = A.dot(x)

        t_min, t_max = -np.inf, np.inf
        for i in range(len(b)):
            if abs(Ad[i]) < 1e-12:
                if Ax[i] > b[i]:
                    t_min, t_max = 1, -1  # Infeasible, force no move
                    break
                continue
            t_i = (b[i] - Ax[i]) / Ad[i]
            if Ad[i] > 0:
                t_max = min(t_max, t_i)
            else:
                t_min = max(t_min, t_i)

        if t_min <= t_max:
            t = np.random.uniform(t_min, t_max)
            x = x + t * d

        if step >= burn_in and (step - burn_in) % steps_per_sample == 0:
            samples.append(x.copy())

    return np.array(samples)


# Example usage
if __name__ == "__main__":
    horizon = 18
    skewness = 10
    l_bounds = (1.0, 60.0)

    A, b = build_polytope(horizon, skewness, l_bounds=l_bounds, l0=10)
    x0 = np.array([10] * horizon)
    samples = hit_and_run(A, b, x0, n_samples=10)
    
    qn: ClosedQueuingNetwork = acmeair_qn()
    qn.set_controllers(
        [constant_controller(qn, 0, qn.max_users)] +
        autoscalers['hpa50'](qn)
    )

    for sample in samples:
        c_init = [1] * (qn.stations - 1)
        q, s, d, c = qn.steady_state_simulation(c_init, sample, 3)
        rtv = qn.compute_rtv(sample, s[:,1:])
        print(rtv, sample)
        