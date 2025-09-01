import os
import numpy as np
from libs.qn.examples.closed_queuing_network import random_qn
from libs.qn.examples.controller import constant_controller
from libs.qn.model.queuing_network import ClosedQueuingNetwork
from libs.qn.examples.controller import autoscalers

FOLDER = 'resources/random_qns/'

qn_filter = lambda qn: np.dot(qn.visit_vector, 1/qn.mu[1:]) < 0.5 and all(qn.visit_vector != 0)

os.makedirs(FOLDER, exist_ok=True)

i = 0
while i < 50:
    qn: ClosedQueuingNetwork = random_qn(3)
    qn.set_controllers(
        [constant_controller(qn, 0, qn.max_users)] +
        autoscalers['hpa50'](qn)
    )

    if qn_filter(qn):
        print(f"QN {i}\n")
        qn.save(f"{FOLDER}qn_{i}.pkl")
        i += 1 