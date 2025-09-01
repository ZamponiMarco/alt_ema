import json
import os
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FOLDER = 'resources/pics'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TRAJECTORY_FOLDER = 'resources/workloads'
load = json.load(open(os.path.join(TRAJECTORY_FOLDER, 'test.json')))
TRAJECTORY = np.array(load['users'])

x = np.arange(0, len(TRAJECTORY))

plt.figure(figsize=(6, 3.5))
plt.step(x, TRAJECTORY, where='post', linestyle='-', color='navy')
plt.xlabel('$t$', fontsize=12)
plt.ylabel('$l(t)$', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'plot_load_acme.pdf'), bbox_inches='tight')