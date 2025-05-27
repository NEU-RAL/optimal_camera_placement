import json
import os

# # Reload runtime data from the files
# runtime_data = {}
# directory = "."
# for file in os.listdir(directory):
#     if file.startswith("results_n") and file.endswith(".json"):
#         with open(os.path.join(directory, file), "r") as f:
#             data = json.load(f)
#             n = data["params"]["n"]
#             stats = data.get("stats", {})
#             if "continuous_relaxation" in stats and "greedy" in stats:
#                 runtime_data[n] = {
#                     "fw": {
#                         "mean": stats["continuous_relaxation"]["time_mean"],
#                         "std": stats["continuous_relaxation"]["time_std"],
#                     },
#                     "greedy": {
#                         "mean": stats["greedy"]["time_mean"],
#                         "std": stats["greedy"]["time_std"],
#                     }
#                 }

# # Now regenerate the dual-scale plot with updated runtime_data
# import matplotlib.pyplot as plt
# import numpy as np

# n_values = np.array(sorted(runtime_data.keys()))
# fw_means = np.array([runtime_data[n]['fw']['mean'] for n in n_values])
# fw_stds = np.array([runtime_data[n]['fw']['std'] for n in n_values])
# greedy_means = np.array([runtime_data[n]['greedy']['mean'] for n in n_values])
# greedy_stds = np.array([runtime_data[n]['greedy']['std'] for n in n_values])

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

# def plot_with_errorbars(ax, yscale):
#     ax.errorbar(n_values, fw_means, yerr=fw_stds, label='Convex Relaxation', marker='o', linestyle='-', capsize=4)
#     ax.errorbar(n_values, greedy_means, yerr=greedy_stds, label='Greedy', marker='s', linestyle='-', capsize=4)
#     ax.set_xlabel('Number of Variables (n)', fontsize=12)
#     ax.set_ylabel('Runtime (seconds)', fontsize=12)
#     ax.set_yscale(yscale)
#     ax.set_title(f'Runtime Comparison ({yscale.capitalize()} Scale)', fontsize=13)
#     ax.grid(True, linestyle='--', alpha=0.6)
#     ax.legend(fontsize=10)
#     ax.tick_params(axis='both', which='major', labelsize=10)

# plot_with_errorbars(ax1, 'linear')
# plot_with_errorbars(ax2, 'log')

# plt.tight_layout()
# plt.subplots_adjust(wspace=0.3)
# plt.show()
