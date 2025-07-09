import time
import os
from collections import Counter

# -- Reading modules ---
from reader.reader import MIPInstance
# --- Pre-solver modules ---
from presolve.ModelCanonicalizer import ModelCanonicalizer
from presolve.engine import PresolveEngine
from presolve.clean_model import CleanModel
from presolve.singleton_cols import ColSingletonRemover
from presolve.bound_tightening import BoundTightener
from presolve.coeff_tightening import CoefficientTightening
from presolve.dual_fix import DualFix
#--- Solver modules ---
from bnb.solver import BranchAndBoundSolver


# --- Pre-Solver function ---
def run_presolve(instance: MIPInstance):
    """
    Runs the full presolve pipeline on the given instance,
    modifying it in place
    """
    ModelCanonicalizer().apply(instance)
    engine = PresolveEngine(instance)
    engine.register(CleanModel())
    engine.register(ColSingletonRemover())
    engine.register(BoundTightener())
    engine.register(CoefficientTightening())
    engine.register(DualFix())
    engine.run()
    # engine.summary()
    return engine.applied_reductions


# --- Solver function ---
def run_solver(instance: MIPInstance):
    """
    Initializes and runs the custom Branch-and-Bound solver on the given instance
    """
    #Configure your solver here as needed
    solver = BranchAndBoundSolver(instance,
                                  enable_plunging=True,
                                  k_plunging=11,
                                  enable_pump=True,
                                  n_pump=1,
                                  fp_max_it=100
    )

    solution, obj_value = solver.solve()
    return solution, obj_value


# --- Other utilities ---

def summarize_reductions(applied_reductions):
    """
    Prints a summary of reductions applied
    """
    counter = Counter(r.kind for r in applied_reductions)
    print("\nReduction Summary:")
    print(f"{'Reduction Type':<25} | Count")
    print("-" * 40)
    for kind, count in counter.items():
        print(f"{kind:<25} | {count}")
    print("-" * 35)

def print_summary_comparison(stats_before,stats_after):
    """ Prints a before-and-after table of model statistics"""
    print("\n📊 Model Statistics Comparison:")
    print(f"{'':<20} | {'Original':>12} | {'Presolved':>12}")
    print("-" * 50)
    print(f"{'Constraints':<18} | {stats_before['cons']:>12} | {stats_after['cons']:>12}")
    print(f"{'Total Variables':<18} | {stats_before['vars']:>12} | {stats_after['vars']:>12}")
    print(f"{'  - Binary':<18} | {stats_before['bin_vars']:>12} | {stats_after['bin_vars']:>12}")
    print(f"{'  - Integer':<18} | {stats_before['int_vars']:>12} | {stats_after['int_vars']:>12}")
    print(f"{'  - Continuous':<18} | {stats_before['cont_vars']:>12} | {stats_after['cont_vars']:>12}")
    print("-" * 50)

def get_stats(instance: MIPInstance)->dict:
    """
    Extracts key statistics from an MIPInstance object
    """
    return {
        "cons": instance.num_constraints,
        "vars": instance.num_vars,
        "bin_vars":instance.num_binary,
        "int_vars": instance.num_integer,
        "cont_vars": instance.num_continuous
    }

# --- Main execution block ---

def main():
    # 1. SET-UP
    #Path to the .mps file
    instance_name="instance_0017.mps"
    instance_path = os.path.join("Test_instances", instance_name)  # Adjust directory if needed

    # 2. LOAD
    print(f"📦 Loading instance: {instance_name}")
    instance = MIPInstance(instance_path)
    stats_before = get_stats(instance)

    #3. PRESOLVE
    start_pre_solver_time=time.time()
    applied_reductions = run_presolve(instance)
    total_pre_solver_time=time.time()-start_pre_solver_time
    stats_after=get_stats(instance)
    print(f"Pre-solve complete in {total_pre_solver_time:.2f} seconds")

    #4. REPORT
    print_summary_comparison(stats_before, stats_after)
    summarize_reductions(applied_reductions)

    #5. SOLVE
    print("Solving with Branch-and-Bound (solver created by Lucas Kuhnen...)")
    solution,obj_value=run_solver(instance)

    # Print results
    if solution is None:
        print("\n❌ No feasible solution found.")
    else:
        print("\n✅ Optimal solution found!")
        print(f"Objective value: {obj_value:.6f}")

if __name__ == "__main__":
    main()





