import time
import argparse
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
    engine.summary()
    return engine.applied_reductions


# --- Solver function ---

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
    TEST_INSTANCE_FOLDER = "Test_instances"
    parser = argparse.ArgumentParser(description="A custom MIP solver.")
    parser.add_argument("instance_filename", type=str,
                        help=f"Filename of the .mps instance in the '{TEST_INSTANCE_FOLDER}' folder.")
    parser.add_argument("--no-presolve", action="store_true", help="Disable the presolve step.")
    parser.add_argument("--no-cuts", action="store_true", help="Disable clique cuts.")  # ⬅️ ADD THIS
    args = parser.parse_args()

    # 3. Construct the full path from the folder and filename
    full_path = os.path.join(TEST_INSTANCE_FOLDER, args.instance_filename)

    # 2. LOAD
    # All subsequent code now uses the `full_path` variable
    print(f"📦 Loading instance: {full_path}")
    if not os.path.exists(full_path):
        print(f"❌ ERROR: File not found at '{full_path}'")
        return
    instance = MIPInstance(full_path)
    instance.pretty_print()
    stats_before = get_stats(instance)

    #3. PRESOLVE

    if not args.no_presolve:
        print("⚙️  Starting pre-solver...")
        start_pre_solver_time = time.time()
        applied_reductions = run_presolve(instance)
        total_pre_solver_time = time.time() - start_pre_solver_time
        stats_after = get_stats(instance)
        print(f"Pre-solve complete in {total_pre_solver_time:.2f} seconds")
        print_summary_comparison(stats_before, stats_after)
        # summarize_reductions(applied_reductions)
    else:
        print("⏩ Skipping presolve step as requested.")

    #5. SOLVE
    print("Solving with Branch-and-Bound (solver created by Lucas Kuhnen...)")
    solver = BranchAndBoundSolver(instance,
                                  enable_plunging=True,
                                  k_plunging=10,
                                  enable_pump=True,
                                  n_pump=2,
                                  fp_max_it=20000,
                                  clique_cuts=not args.no_cuts)  # ⬅️ PASS THE ARGUMENT HERE

    solution, obj_value = solver.solve()

    # Print results
    if solution is None:
        print("\n❌ No feasible solution found.")
    else:
        # print("\n✅ Optimal solution found!")
        print(f"Objective value: {obj_value:.6f}")

if __name__ == "__main__":
    main()





