import os
import matplotlib.pyplot as plt
from reader.reader import MIPInstance
from bnb.solver import BranchAndBoundSolver
from main import run_presolve


def run_and_get_history(instance_path, depth_test,strong_k_test):
    print(f"\n--- Running: {os.path.basename(instance_path)} (Depth: {depth_test}) (Strong K: {strong_k_test}) ---")
    instance = MIPInstance(instance_path)
    run_presolve(instance)
    solver = BranchAndBoundSolver(instance,
                                  enable_plunging=True,
                                  k_plunging=10,
                                  clique_cuts=True,
                                  strong_depth=depth_test,
                                  strong_k=strong_k_test)
    solution, obj_value, times, primal_bounds, dual_bounds = solver.solve()

    return times, primal_bounds, dual_bounds


if __name__ == "__main__":
    # --- Configuration ---
    INSTANCE_FOLDER = "Test_instances"
    # INSTANCE_FILENAME = "model_S1_Jc0_Js9_T96.mps"

    INSTANCE_FILENAME="instance_0012.mps"

    # Define the parameter grid for the experiment
    strong_depths = [10, 50, 100]
    strong_ks = [20, 200, 2000]
    linestyles = [':', '--', '-']  # Dotted, dashed, and solid lines for k values

    # Create a 3x3 grid of subplots
    # Correctly create 3 rows and 1 column
    fig, axes = plt.subplots(len(strong_depths), 1, figsize=(12, 24), sharex=True)
    fig.suptitle(f'Strong Branching Parameter Analysis for {INSTANCE_FILENAME}', fontsize=20)

    for i, depth in enumerate(strong_depths):
        ax = axes[i]
        ax.set_title(f"strong_depth = {depth}", fontsize=14)
        instance_path = os.path.join(INSTANCE_FOLDER, INSTANCE_FILENAME)

        for j, k_val in enumerate(strong_ks):
            depth_test = depth
            strong_k_test = k_val

            # Run the solver and get the history
            times, primals, duals = run_and_get_history(instance_path, depth_test, strong_k_test)


            if times:
                style=linestyles[j]

                ax.step(times, primals, where='post', label=f'Primal Bound (k={k_val})', linestyle=style, color='blue')
                ax.step(times, duals, where='post', label=f'Dual Bound (k={k_val})',linestyle=style, color='red')

            ax.set_ylabel("Objective Value")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

    plt.tight_layout(rect=[0.1, 0.1, 1, 0.95])
    plot_filename = f"strong_branching_grid_{INSTANCE_FILENAME.replace('.mps', '')}.png"
    plt.savefig(plot_filename)
    print(f"\n📈 Experiment complete. Grid plot saved to {plot_filename}")