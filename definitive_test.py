import os
import glob
import csv
import time
import matplotlib.pyplot as plt
import numpy as np

# Import your project components
from reader.reader import MIPInstance
from bnb.solver import BranchAndBoundSolver
from main import run_presolve  # Assuming this is your presolve function


def generate_plot(times, primals, duals, instance_name, output_folder):
    plt.figure(figsize=(10, 6))
    plt.step(times, primals, where='post', label='Primal Bound', color='blue')
    plt.step(times, duals, where='post', label='Dual Bound', color='red')

    plt.xlabel("Time (s)")
    plt.ylabel("Objective Value")
    plt.title(f"Convergence: {instance_name}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plot_filename = f"{instance_name.replace('.mps', '')}.png"
    plt.savefig(os.path.join(output_folder, plot_filename))
    plt.close()


def main():
    INSTANCE_FOLDER = "Test_instances"
    PLOT_FOLDER = "output_figures"
    RESULTS_CSV = "experiment_results.csv"
    os.makedirs(PLOT_FOLDER, exist_ok=True)
    instance_paths = glob.glob(os.path.join(INSTANCE_FOLDER, '*.mps'))

    # --- Open CSV file and write header ---
    with open(RESULTS_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Instance', 'Objective', 'Time', 'Nodes'])

        # --- Main Experiment Loop ---
        for path in sorted(instance_paths):
            instance_name = os.path.basename(path)
            print(f"\n{'=' * 80}\nRunning {instance_name}\n{'=' * 80}")
            instance = MIPInstance(path)
            instance.pretty_print()
            run_presolve(instance)
            solver = BranchAndBoundSolver(instance,
                                          enable_plunging=True,
                                          k_plunging=10,
                                          clique_cuts=False,
                                          strong_depth=100,
                                          strong_k=20)
            sol, obj_value, times, primals, duals, nodes_explored, runtime = solver.solve()
            writer.writerow([instance_name, obj_value, runtime, nodes_explored])
            # if times:
            #     generate_plot(times, primals, duals, instance_name, PLOT_FOLDER)

    print("\n\n✅ All tests complete. Results saved to experiment_results.csv")


if __name__ == "__main__":
    main()