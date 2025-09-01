import json
import os

import numpy as np


BASE_FOLDER = 'resources/workloads'

if __name__ == '__main__':
    trace = 0
    times = np.array([])
    
    for load in os.listdir(BASE_FOLDER):
        with open(os.path.join(BASE_FOLDER, load)) as f:
            data = json.load(f)
            time = data['time']
            users = data['users']
            status = data['status']
            
            if len(users) == 0:
                trace += 1
                
            times = np.append(times, time)
            
    print(f'Minimum time: {times.min()}')
    print(f'Maximum time: {times.max()}')
    print(f'Mean time: {times.mean()}')
    print(f'Standard deviation of time: {times.std()}')
    print(f'Number of traces: {len(times)}')
    print(f'Number of traces with no users: {trace}')
    print(f'Number of traces with users: {len(times) - trace}')      