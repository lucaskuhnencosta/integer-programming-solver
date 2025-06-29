import os
from reader.reader import MIPInstance
from bnb.solver import BranchAndBoundSolver

def main():
    # Path to your .mps file (change this to any instance you want to test)
    instance_name = "instance_0001.mps"
    instance_path = os.path.join("Test_instances", instance_name)

    # Load the instance
    print(f"📦 Loading instance: {instance_name}")
    instance = MIPInstance(instance_path)

    # Run the Branch-and-Bound solver
    print("🚀 Solving with Branch-and-Bound...")
    solver = BranchAndBoundSolver(instance,enable_plunging=True,k_plunging=200,enable_pump=True,n_pump=50,fp_max_it=1000)
    solution, obj_value = solver.solve()

    # Print results
    if solution is None:
        print("\n❌ No feasible solution found.")
    else:
        print("\n✅ Optimal solution found!")
        print(f"Objective value: {obj_value:.6f}")

if __name__ == "__main__":
    main()
