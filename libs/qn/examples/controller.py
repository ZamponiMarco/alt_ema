import numpy as np

from libs.qn.model.queuing_network import ClosedQueuingNetwork
from libs.qn.pwa.pwa_function import PWAFunction, Region
from pycvxset import Polytope


def cpu_hpa(network: ClosedQueuingNetwork, tracked_load: float, station: int) -> PWAFunction:
    upper_cores = network.max_cores[station]
    lower_cores = network.min_cores[station]
    upper_disturbance = network.max_disturbance[station]
    mu = network.mu[station]

    poly = Polytope(lb=[lower_cores, 0.0], ub=[upper_cores, upper_disturbance])
    A = np.array([0, 1 / mu * 1/tracked_load])
    b = 0

    region = Region(poly, A, b)
    return PWAFunction(
        regions=[region],
        value_bounds=(lower_cores, upper_cores)  # Bounds for the value
    )

def constant_controller(network: ClosedQueuingNetwork, station: int, value: float) -> PWAFunction:
    upper_cores = network.max_cores[station]
    lower_cores = network.min_cores[station]
    upper_disturbance = network.max_disturbance[station]

    poly = Polytope(lb=[lower_cores, 0], ub=[upper_cores, upper_disturbance])
    A = np.array([0, 0])
    b = value

    region = Region(poly, A, b)
    return PWAFunction(
        regions=[region],
        value_bounds=(lower_cores, upper_cores)  # Bounds for the value
    )

def cpu_threshold_controller(network: ClosedQueuingNetwork, station: int, lower: float, upper: float, magnitude: int) -> PWAFunction:
    upper_cores = network.max_cores[station]
    lower_cores = network.min_cores[station]
    upper_disturbance = network.max_disturbance[station]
    mu = network.mu[station]

    poly_base = Polytope(
        lb=[lower_cores, 0], ub=[upper_cores, upper_disturbance]
        )
    
    poly_decrease = poly_base.intersection_with_halfspaces(
        A=np.array([- lower, 1/mu]), b=np.array([0])
    )
    A_decrease = np.array([1, 0])
    b_decrease = -magnitude

    region_decrease = Region(poly_decrease, A_decrease, b_decrease)
    
    poly_increase = poly_base.intersection_with_halfspaces(
        A=np.array([upper, -1/mu]), b=np.array([0])
    )
    A_increase = np.array([1, 0])
    b_increase = magnitude
    
    region_increase = Region(poly_increase, A_increase, b_increase)
    
    poly_stay = poly_base.intersection_with_halfspaces(
        A=np.array([
            [-upper, 1/mu], 
            [lower, -1/mu]
            ]), 
        b=np.array([0, 0])
    )
    A_stay = np.array([1, 0])
    b_stay = 0
    
    region_stay = Region(poly_stay, A_stay, b_stay)
    
    return PWAFunction(
        regions=[region_decrease, region_increase, region_stay],
        value_bounds=(lower_cores, upper_cores)  # Bounds for the value
    )
    
def cpu_step_controller(network: ClosedQueuingNetwork, station: int, steps: list) -> PWAFunction:
    upper_cores = network.max_cores[station]
    lower_cores = network.min_cores[station]
    upper_disturbance = network.max_disturbance[station]
    mu = network.mu[station]
    
    regions = []
    
    for step in steps:
        lb = step[0]
        ub = step[1]
        coef = step[2]
        b = step[3]
        
        poly = Polytope(
            lb=[lower_cores, 0], ub=[upper_cores, upper_disturbance]
        )
        poly = poly.intersection_with_halfspaces(
            A=np.array([[lb, -1/mu], [-ub, 1/mu]]), b=np.array([0, 0])
        )
        A = np.array([coef, 0])
        b = b
        
        region = Region(poly, A, b)
        regions.append(region)
        
    return PWAFunction(
        regions=regions,
        value_bounds=(lower_cores, upper_cores)  # Bounds for the value
    )


step_2 = [
    [0.0, 0.2, 1,-2],
    [0.2, 0.4, 1, -1],
    [0.4, 0.6, 1, 0],
    [0.6, 0.8, 1, 1],
    [0.8, 1.0, 1, 2]
]
autoscalers = {
    'hpa50': lambda network: [cpu_hpa(network, 0.5, station) for station in range(1, network.stations)],
    'hpa60': lambda network: [cpu_hpa(network, 0.6, station) for station in range(1, network.stations)],
    'step1': lambda network: [cpu_threshold_controller(network, station, 0.3, 0.7, 1) for station in range(1, network.stations)],
    'step2': lambda network: [cpu_step_controller(network, station, step_2) for station in range(1, network.stations)],
}