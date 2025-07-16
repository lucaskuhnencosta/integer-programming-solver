import os
import matplotlib.pyplot as plt
import numpy as np

# Import your solver components
from reader.reader import MIPInstance
from bnb.solver import BranchAndBoundSolver
from main import run_presolve  # Or however your presolve is called
from presolve.ModelCanonicalizer import ModelCanonicalizer

def run_and_get_history(instance_path, clique_cuts_test):
    print(f"\n--- Running: {os.path.basename(instance_path)} (Cliques: {clique_cuts_test}) ---")
    instance = MIPInstance(instance_path)
    run_presolve(instance)  # Assuming you have a presolve function
    solver = BranchAndBoundSolver(instance,
                                  enable_plunging=True,
                                  k_plunging=10,
                                  clique_cuts=clique_cuts_test,
                                  strong_depth=10,
                                  strong_k=1500)
    solution, obj_value, times, primal_bounds, dual_bounds = solver.solve()

    return times, primal_bounds, dual_bounds

if __name__ == "__main__":

    INSTANCE_FOLDER = "Test_instances"
    instances_to_test = ["instance_0016.mps","model_S1_Jc0_Js11_T96.mps"]
    # solver_params = {
    #     'enable_plunging': True,
    #     'k_plunging': 10,
    #     'clique_cuts': True,
    #     'strong_depth': 10,
    #     'strong_k': 1500
    # }

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Effect of clique cuts', fontsize=16)

    for i, filename in enumerate(instances_to_test):
        ax=axes[i]
        instance_path=os.path.join(INSTANCE_FOLDER, filename)

        t_pre, p_pre, d_pre = run_and_get_history(instance_path, True)

        # Run without Presolve
        t_no_pre, p_no_pre, d_no_pre = run_and_get_history(instance_path, False)

        # --- Plotting ---
        # With Presolve
        ax.step(t_pre, p_pre, where='post', label='Primal Bound (Presolve)', color='blue', linestyle='-')
        ax.step(t_pre, d_pre, where='post', label='Dual Bound (Presolve)', color='red', linestyle='-')

        # Without Presolve
        ax.step(t_no_pre, p_no_pre, where='post', label='Primal Bound (No Presolve)',color='blue', linestyle='--')
        ax.step(t_no_pre, d_no_pre, where='post', label='Dual Bound (No Presolve)', color='red', linestyle='--')

        ax.set_title(f"Instance: {filename}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Objective Value")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # --- Save and Show ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    plot_filename = "convergence_comparison_2.png"
    plt.savefig(plot_filename)

    print(f"\n📈 Comparison plot saved to {plot_filename}")
    plt.show()