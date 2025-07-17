import gurobipy as gp
from gurobipy import GRB
import sys

# --- CONFIGURATION ---
# The path to the .mps file you want to solve.
# You can change this or pass it as a command-line argument.
MPS_FILE_PATH = "reader/amodel_S1_Jc0_Js13_T96.mps"

def solve_with_gurobi(file_path):
    """Reads and solves a MIP from an .mps file using Gurobi."""
    try:
        # 1. Read the .mps file into a Gurobi model
        print(f"📦 Reading instance: {file_path}")
        model = gp.read(file_path)

        # 2. Solve the model
        print("🚀 Solving with Gurobi...")
        model.optimize()

        # 3. Check and print the results
        print("\n" + "="*30)
        print("          RESULTS          ")
        print("="*30)

        if model.Status == GRB.OPTIMAL:
            print(f"✅ Optimal solution found!")
            print(f"   - Objective Value: {model.ObjVal:.6f}")
            print(f"   - Best Bound:      {model.ObjBound:.6f}")
            print(f"   - Optimality Gap:  {model.MIPGap*100:.4f}%")
            print(f"   - Solve Time:      {model.Runtime:.4f} seconds")
        elif model.Status == GRB.INFEASIBLE:
            print("❌ Model is infeasible.")
        elif model.Status == GRB.UNBOUNDED:
            print("❌ Model is unbounded.")
        else:
            print(f"Solver finished with status code: {model.Status}")
            if model.SolCount > 0:
                 print(f"   - Best solution found: {model.ObjVal:.6f}")


    except gp.GurobiError as e:
        print(f"Gurobi error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Use the file path from the command line if provided, otherwise use the default.
    if len(sys.argv) > 1:
        solve_with_gurobi(sys.argv[1])
    else:
        solve_with_gurobi(MPS_FILE_PATH)