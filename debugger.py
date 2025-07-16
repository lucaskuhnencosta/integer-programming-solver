import gurobipy as gp
from gurobipy import GRB
import os
import argparse

# Import your solver components
from reader.reader import MIPInstance
from main import run_presolve  # Or however your presolve is called


def solve_with_gurobi(file_path):
    """Helper function to solve an MPS file and return the objective value."""
    try:
        model = gp.read(file_path)
        model.optimize()
        if model.Status == GRB.OPTIMAL:
            return model.ObjVal
        else:
            print(f"Gurobi could not solve {file_path} to optimality.")
            return None
    except gp.GurobiError as e:
        print(f"Gurobi error on {file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Presolver validation script.")
    parser.add_argument("instance_filename", type=str, help="Filename of the .mps instance.")
    args = parser.parse_args()

    # --- Setup ---
    raw_file_path = os.path.join("Test_instances", args.instance_filename)
    debug_folder = "debugging_original_model_canonical"
    os.makedirs(debug_folder, exist_ok=True)
    presolved_file_path1 = os.path.join(debug_folder, f"presolved_{args.instance_filename}.mps")
    presolved_file_path2 = os.path.join(debug_folder, f"presolved_{args.instance_filename}.lp")

    # --- 1. Solve the original raw model ---
    print(f"--- Solving raw model: {raw_file_path} ---")
    obj_raw = solve_with_gurobi(raw_file_path)
    if obj_raw is None: return

    # --- 2. Load, Presolve, and Save the new model ---
    print("\n--- Running Presolver ---")
    instance = MIPInstance(raw_file_path)
    run_presolve(instance)  # Your presolve function
    instance.write_model(presolved_file_path1)
    instance.write_model(presolved_file_path2)

    # --- 3. Solve the presolved model ---
    print(f"\n--- Solving presolved model: {presolved_file_path1} ---")
    obj_presolved = solve_with_gurobi(presolved_file_path1)
    if obj_presolved is None: return

    # --- 4. Compare the objectives ---
    print("\n" + "=" * 40)
    print("          VALIDATION RESULTS          ")
    print("=" * 40)
    print(f"Original Objective: {obj_raw:.6f}")
    print(f"Presolved Objective: {obj_presolved:.6f}")
    print(f"Presolver Objective Constant: {instance.obj_const:.6f}")

    # The correct check must account for the constant offset from presolving
    final_presolved_value = obj_presolved + instance.obj_const

    if abs(obj_raw - final_presolved_value) < 1e-6:
        print("\n✅ SUCCESS: Presolver correctly preserved the objective value.")
    else:
        print("\n❌ FAIL: Presolver changed the objective value.")
        print(f"   Original: {obj_raw:.6f} vs. Presolved+Constant: {final_presolved_value:.6f}")


if __name__ == "__main__":
    main()