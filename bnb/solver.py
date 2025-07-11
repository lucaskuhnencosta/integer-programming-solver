import time
import random
import numpy as np

from bnb.active_path import ActivePathManager
from bnb.node import Node
from bnb.branching import Branching
from bnb.shared_state import SharedState
from bnb.tree import BranchAndBoundTree

import gurobipy as gp
from gurobipy import GRB

class BranchAndBoundSolver:
    def __init__(self, mip_instance,enable_plunging=False,k_plunging=10,enable_pump=False,n_pump=100,fp_max_it=1000):
        self.instance = mip_instance
        self.shared = SharedState()

        self.brancher = Branching(self.instance)
        self.gap_treshold = 1e-3 #%

        self.enable_plunging = enable_plunging
        self.k_plunging = k_plunging

        self.enable_pump = enable_pump
        self.n_pump = n_pump
        self.fp_max_it=fp_max_it

        self.best_obj=float("inf")
        self.best_sol=None

    def solve(self):
        start_total_solver_time = time.time()

        self.instance.build_root_model() #We will log here
        working_model = self.instance.root_lp_model.copy()
        working_model.setParam("OutputFlag", 0)
        root=Node(parent=None,depth=0,bound_changes={})
        start_time = time.time()
        root.evaluate_lp(working_model, self.instance)
        elapsed_time = time.time() - start_time
        root.node_type = 'root'

        if root.is_infeasible:
            print("\n The problem is infeasible")
            return None,None

        print("\nRoot relaxation: objective %.8f, time %.2f seconds\n" % (root.bound, elapsed_time))

        tree = BranchAndBoundTree()
        active_mgr = ActivePathManager(root,self.instance)
        tree.push(root)
        node_counter = 1
        incumbent = None
        print_stats=False

        mode = "best-first"

        gap_number=None
        global_best_bound = -float('inf')
        previous_best_bound=-float('inf')
        branching_times = []
        incumbent_str="     -     "
        gap_str = "  -  "

        # Gurobi-style column header
        print("    Nodes     |          Current Node         |        Objective Bounds        |  Time   |")
        print(" Expl  Unexpl |  Obj         Depth     IntInf | Incumbent    BestBd      Gap   |   (s)   |")
        print("--------------+-------------------------------+--------------------------------+----------+")

        try:
            while not tree.empty():
                current_tree_best_bound = tree.get_best_bound()
                if mode == "dfs":
                    node=tree.pop_dfs() #LIFO
                    if node is None: #DFS path is exhausted
                        mode="best-first"
                        continue
                else: #best-first
                    node=tree.pop_best_bound()
                    if node is None:
                        break

                node_counter += 1
                active_mgr.switch_focus(node, working_model)
                start_time=time.time()
                node.evaluate_lp(working_model,self.instance)
                elapsed_time=time.time() - start_time

                unexplored = len(tree)

                if node.is_infeasible or node.bound >= self.best_obj:
                    self.prune_if_leaf(node)
                    continue

                int_infeas = sum(
                    1 for i, v in enumerate(node.solution or [])
                    if self.instance.var_types[i] in ['B', 'I'] and abs(v - round(v)) > 1e-6
                )

                if self.instance.is_integral(node.solution):
                    current_obj = node.bound + self.instance.obj_const
                    if node.bound < self.best_obj:
                        self.best_obj=node.bound
                        self.best_sol=node.solution
                        incumbent=self.best_obj
                        incumbent_str = f"{incumbent:10.5f}"
                        gap_number=(incumbent - global_best_bound) / abs(incumbent)
                        gap_str = f"{gap_number*100:4.1f}%"
                        print(f"P{node_counter:5d} {unexplored:5d}  |{node.bound:8.5f}  {node.depth:5d}      {int_infeas:6d}  |{incumbent_str}  {global_best_bound:8.5f}  {gap_str}  |  {elapsed_time:4.2f}s")
                        if incumbent and gap_number and gap_number<self.gap_treshold:
                            break
                    self.prune_if_leaf(node)
                    node_counter += 1
                    continue

                if not incumbent and self.enable_pump and (node_counter%self.n_pump==0):
                    if node.solution is not None:
                        fp_obj,fp_sol=self.feasibility_pump(node.solution, working_model)
                        if fp_obj is not None and fp_obj<self.best_obj:
                            self.best_obj = fp_obj
                            self.best_sol = fp_sol
                            incumbent = self.best_obj
                            incumbent_str = f"{incumbent:10.5f}"
                            print(f"FP FOUND NEW INCUMBENT: {self.best_obj:.5f}")


                global_best_bound = max(global_best_bound, current_tree_best_bound)

                if incumbent:
                    gap_number = (incumbent - global_best_bound) / abs(incumbent)
                    gap_str = f"{gap_number * 100:4.1f}%"

                if global_best_bound > previous_best_bound:
                    print(f"D{node_counter:5d} {unexplored:5d}  |   -      {node.depth:5d}      {int_infeas:6d}  |{incumbent_str}  {global_best_bound:8.5f}  {gap_str}  |  {elapsed_time:4.2f}s")
                previous_best_bound=global_best_bound

                # Select branching variable using strong branching (or other strategy)

                branch_start = time.time()
                branch_var, left_node, right_node = self.brancher.select_branching_variable(node,node.solution,working_model,active_mgr)
                branch_end = time.time()
                branching_times.append(branch_end - branch_start)
                if branch_var is None:
                    self.prune_if_leaf(node)
                    continue # No valid branching var found

                node.children.extend([left_node, right_node])
                node.node_type = 'fork' if node.lp_basis else 'junction'

                tree.push_children(left_node,right_node)

                if self.enable_plunging and (node_counter>1 and node_counter % self.k_plunging ==0):
                    if mode!="dfs":
                        mode="dfs"

        except KeyboardInterrupt:
            print("\n\n🛑 User interrupt detected! Stopping search.")
            print("Returning best solution found so far.")

        finally:
            end_total_solver_time = time.time()
            elapsed_total_solver_time = end_total_solver_time - start_total_solver_time
            print("Total solver time: {:.4f} seconds".format(elapsed_total_solver_time))
            print(f"Total number of nodes explored:{node_counter}")

            if branching_times:
                avg_branch_time = sum(branching_times) / len(branching_times)
                print(f"\n📊 Average branching time: {avg_branch_time:.4f} seconds")
            else:
                print("\n📊 No branching occurred (possibly solved at root)")

            return self.best_sol, self.best_obj

    import random  # ⬅️ ADD THIS IMPORT AT THE TOP OF YOUR FILE

    import random  # Make sure this import is at the top of your file

    # In your BranchAndBoundSolver class:
    def feasibility_pump(self, start_x, working_model):
        if start_x is None:
            print("Pump skipped: No starting solution provided.")
            return None, None

        # --- (Initial logging and setup is the same) ---
        print("\n\n=======================================================")
        print("🔁 Starting Feasibility Pump...")
        print("=======================================================")
        # ...

        I = [i for i, t in enumerate(self.instance.var_types) if t in ['B', 'I']]
        N = range(self.instance.num_vars)
        x_bar = np.array(start_x)
        previous_distance = float('inf')
        stall_counter = 0

        for iteration in range(self.fp_max_it):
            # ... (Iteration logging, rounding, and perturbation logic is the same) ...

            # --- Step 2: Solve the projection LP ---
            proj_model = gp.Model("fp_projection")
            proj_model.Params.OutputFlag = 0

            # Create variables and keep a reference to them
            x_vars = proj_model.addVars(N, lb=self.instance.lb, ub=self.instance.ub, vtype=GRB.CONTINUOUS, name="x")
            d_vars = proj_model.addVars(I, lb=0.0, name="d")
            proj_model.update()  # Ensure variables are created

            # --- (Constraint and objective setup is the same) ---
            # ...

            proj_model.optimize()

            if proj_model.Status != GRB.OPTIMAL:
                print(f"\n❌ Projection LP failed with status: {proj_model.Status}. Stopping pump.")
                return None, None

            distance = proj_model.ObjVal

            # --- (Logging is the same) ---
            # ...

            # --- FIX: ROBUST SOLUTION RETRIEVAL ---
            if distance < 1e-6:
                print("\n✅ SUCCESS! Feasibility Pump found a feasible solution.")

                # The solution is in the Gurobi 'x' variables, not x_round.
                # Get the solution values robustly.
                sol = np.array(proj_model.getAttr("X", x_vars).values())

                # Double-check that this solution is truly integer-feasible
                if self._count_integer_infeasibilities(sol) > 0:
                    print("   [WARNING] Pump solution has distance=0 but is not integer. Rounding final solution.")
                    for j in I:
                        sol[j] = round(sol[j])

                original_obj_val = np.dot(self.instance.obj, sol) + self.instance.obj_const
                print(f"    Solution Objective: {original_obj_val:.5f}")
                print("=======================================================")
                return original_obj_val, sol

            # Update x_bar for the next iteration
            x_bar = np.array(proj_model.getAttr("X", x_vars).values())
            previous_distance = distance

        print("\n❌ Feasibility Pump did not find a feasible solution after max iterations.")
        print("=======================================================")
        return None, None

    def prune_if_leaf(self, node):
        node.active=False
        parent=node.parent
        while parent:
            parent.children=[c for c in parent.children if c.active]
            if not parent.children:
                parent.active=False
                parent=parent.parent
            else:
                break

    def _check_lp_feasibility(self, sol, tol=1e-6):
        """Checks if a solution vector satisfies all LP constraints."""
        if sol is None:
            return False
        for i in range(self.instance.num_constraints):
            lhs = np.dot(self.instance.A[i, :], sol)
            rhs = self.instance.b[i]
            sense = self.instance.sense[i]
            if sense == 'L' and lhs > rhs + tol:
                return False  # Violation
            elif sense == 'E' and abs(lhs - rhs) > tol:
                return False  # Violation
        return True

    def _count_integer_infeasibilities(self, sol, tol=1e-6):
        """Counts how many integer/binary variables have fractional values."""
        if sol is None:
            return float('inf')
        return sum(1 for i, vtype in enumerate(self.instance.var_types)
                   if vtype in ['B', 'I'] and abs(sol[i] - round(sol[i])) > tol)
