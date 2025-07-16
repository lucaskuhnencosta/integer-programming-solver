import os
import argparse
import gurobipy as gp
from gurobipy import GRB


# Import all of your project's components
from reader.reader import MIPInstance
from presolve.ModelCanonicalizer import ModelCanonicalizer
from presolve.engine import PresolveEngine
from presolve.clean_model import CleanModel
from presolve.singleton_cols import ColSingletonRemover
from presolve.bound_tightening import BoundTightener
from presolve.coeff_tightening import CoefficientTightening
from presolve.dual_fix import DualFix


def solve_with_gurobi_file(file_path):
    """Helper function to solve a model file and return its objective value."""
    model = gp.read(file_path)
    # model.Params.OutputFlag = 0
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        return model.ObjVal
    return None  # Return None if not optimal
    # finally:
    #     if model:
    #         model.dispose()



def debug_presolve(instance_path):
    """
    Applies presolver reductions one by one, validating the objective after each step.
    """
    # --- 1. Get the ground truth objective from the original file ---
    print(f"--- Solving original model to get correct objective ---")
    correct_objective = solve_with_gurobi_file(instance_path)
    if correct_objective is None:
        print("❌ Cannot get baseline solution from original file. Aborting.")
        return
    print(f"✅ Correct Objective: {correct_objective:.6f}\n")

    # --- 2. Set up the instance and presolve engine ---
    instance = MIPInstance(instance_path)
    engine = PresolveEngine(instance)
    engine.register(CleanModel())
    engine.register(ColSingletonRemover())
    engine.register(BoundTightener())
    engine.register(CoefficientTightening())
    engine.register(DualFix())
    debug_folder = "debugging_presolve"
    # step_file = os.path.join(debug_folder, f"step_original.lp")
    # instance.write_model(step_file)

    ModelCanonicalizer().apply(instance)

    # os.makedirs(debug_folder, exist_ok=True)
    # step_file1 = os.path.join(debug_folder, f"step_canonical.lp")
    # step_file2 = os.path.join(debug_folder, f"step_canonical.mps")
    # instance.write_model(step_file1)
    # instance.write_model(step_file2)
    # gurobi_model = instance.build_gurobi_model()


    # step_obj = solve_with_gurobi(gurobi_model)

    # if step_obj is None:
    #     print(f"❌ BUG FOUND: Model became infeasible after this step.")
    #     return

    # --- 3. Create a directory for intermediate model files ---


    print(f"--- Starting Step-by-Step Presolve Validation ---")
    reduction_count = 0
    max_rounds = 10  # Limit presolve rounds
    for round_num in range(max_rounds):
        print(f"\n--- Presolve Round {round_num + 1} ---")
        reductions_found_in_round = False
        for presolver in engine.presolvers:
            print(f"\nWe are in presolver {presolver.name}:")
            reductions = presolver.apply(instance)
            if not reductions:
                print("No reductions found\n")
                continue
            reductions_found_in_round = True
            for r in reductions:
                reduction_count += 1
                print("Found following reduction:")
                print(f"[PreSolver {presolver.name}] Applying reduction: {r}")

            engine._apply_reductions(reductions)


            step_obj=None
            model = instance.build_gurobi_model()
            model.optimize()
            if model.Status == GRB.OPTIMAL:
                step_obj=model.ObjVal
            if step_obj is None:
                print(f"❌ BUG FOUND: Model became infeasible after this step.")
                return


            # Validate the objective, accounting for the presolver's constant
            current_total_obj = step_obj + instance.obj_const

            if abs(correct_objective) - abs(current_total_obj) > 1e-5:
                print("\n" + "!" * 60)
                print("🚨 BUG FOUND: Objective value changed after this reduction!")
                print(f"   - Faulty Reduction: {r}")
                print(f"   - Original Objective:   {correct_objective:.6f}")
                print(f"   - New Objective:        {step_obj:.6f}")
                print(f"   - Objective Constant:   {instance.obj_const:.6f}")
                print(f"   - New Total Value:      {current_total_obj:.6f}")
                print(f"   - Intermediate model saved to: {step_file}")
                print("!" * 60)
                step_file1 = os.path.join(debug_folder, f"step_{reduction_count}.lp")
                instance.write_model(step_file1)


                return  # Stop the debugger

            print("We survived until here, lets go to the next pre-solver...")
        print("We survived until here, lets go to the next round...")
        # If no presolver made any changes in a full round, stop
        if not reductions_found_in_round:
            print("\n--- Presolve stabilized. No more reductions found. ---")
            break

    print("\n✅ SUCCESS: All presolve steps completed without changing the optimal objective.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step-by-step presolver debugger.")
    parser.add_argument("instance_filename", type=str, help="Filename of the .mps instance.")
    args = parser.parse_args()

    instance_path = os.path.join("Test_instances", args.instance_filename)
    debug_presolve(instance_path)