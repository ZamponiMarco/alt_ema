import json
import os
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FOLDER = 'resources/pics'
TRAJECTORY_FOLDER = 'resources/workloads'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def load_all_trajectories(folder):
	names = []
	series = []
	for fname in sorted(os.listdir(folder)):
		if not fname.endswith('.json'):
			continue
		fpath = os.path.join(folder, fname)
		try:
			with open(fpath) as f:
				data = json.load(f)
			users = data.get('users')
			if users is None:
				continue
			arr = np.array(users, dtype=float)
			if arr.size == 0:
				continue
			names.append(os.path.splitext(fname)[0])
			series.append(arr)
		except Exception:
			continue
	return names, series


def align_series(series_list):
	if not series_list:
		return np.empty((0, 0))
	max_len = max(len(s) for s in series_list)
	aligned = np.zeros((len(series_list), max_len))
	for i, s in enumerate(series_list):
		aligned[i, : len(s)] = s
		if len(s) < max_len:
			aligned[i, len(s) :] = s[-1]
	return aligned


if __name__ == '__main__':
	names, series = load_all_trajectories(TRAJECTORY_FOLDER)
	if len(series) == 0:
		print('No workload files with key "users" found in', TRAJECTORY_FOLDER)
		raise SystemExit(0)

	aligned = align_series(series)
	x = np.arange(aligned.shape[1] + 1)

	plt.figure(figsize=(6, 3.5))

	# Load test.json for comparison (non-perturbed model)
	test_path = os.path.join(TRAJECTORY_FOLDER, 'test.json')
	test_trajectory = None
	if os.path.isfile(test_path):
		try:
			with open(test_path) as f:
				test_data = json.load(f)
			test_users = test_data.get('users')
			if test_users is not None:
				test_trajectory = np.array(test_users, dtype=float)
		except Exception:
			pass

	# Plot test.json in black for comparison
	if test_trajectory is not None and test_trajectory.size > 0:
		test_ext = np.concatenate([test_trajectory, [test_trajectory[-1]]])
		x_test = np.arange(len(test_ext))
		plt.step(x_test, test_ext, where='post', color='black', label='Nominal', linestyle='--')

	# Summary statistics (mean and std. dev.)
	mean = aligned.mean(axis=0)
	std = aligned.std(axis=0)

	# Extend stats with last sample for step plotting
	mean_ext = np.concatenate([mean, [mean[-1]]])
	std_ext = np.concatenate([std, [std[-1]]])

	plt.step(x, mean_ext, where='post', color='crimson', label='Avg.')
	plt.fill_between(x, mean_ext-std_ext, mean_ext+std_ext, step='post', color='crimson', alpha=0.15, label='Std. Dev.')

	plt.xlabel('$t$', fontsize=12)
	plt.ylabel('$l(t)$', fontsize=12)
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.legend(fontsize=8)
	plt.xlim(-0.5, 30.5)
	plt.ylim(-0.5, mean_ext.max() * 1.05)
	plt.tight_layout()
	plt.savefig(os.path.join(OUTPUT_FOLDER, 'plot_loads_overview.pdf'), bbox_inches='tight', dpi=300)