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
                    # mode="best-first"
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
        # Failsafe guard clause
        if start_x is None:
            # print("Pump skipped: No starting solution provided.")
            return None, None
        #
        # print("\n\n=======================================================")
        # print("🔁 Starting Feasibility Pump (with cycle breaking)...")
        # print("=======================================================")

        # --- Initial State Logging ---
        start_obj = np.dot(self.instance.obj, start_x)
        start_int_infeas = self._count_integer_infeasibilities(start_x)
        start_is_feasible = self._check_lp_feasibility(start_x)
        # print("--- Initial State ---")
        # print(f"  Incoming Solution Obj: {start_obj:.4f}")
        # print(f"  Integer Infeasibilities: {start_int_infeas}")
        # print(f"  Is LP Feasible: {start_is_feasible}")

        I = [i for i, t in enumerate(self.instance.var_types) if t in ['B', 'I']]
        N = range(self.instance.num_vars)
        x_bar = np.array(start_x)
        previous_distance = float('inf')
        stall_counter=0
        for iteration in range(self.fp_max_it):
            # print(f"\n--- Feasibility Pump Iteration {iteration + 1} ---")

            # Step 1: Round x_bar
            x_round = x_bar.copy()
            for j in I:
                x_round[j] = round(x_bar[j])

            rounded_int_infeas = self._count_integer_infeasibilities(x_round)
            # print(f"  After Rounding: Integer Infeasibilities: {rounded_int_infeas} (should be 0)")

            # --- Cycle-Breaking Logic ---
            if 'distance' in locals() and distance >= previous_distance:
                stall_counter+=1
                print("  ⚠️ Cycle detected! Applying random perturbation...")

                if stall_counter>20:
                    return None, None
                troubled_vars = [j for j in I if abs(x_bar[j] - x_round[j]) > 1e-6]
                if not troubled_vars: troubled_vars = I


                num_flips = min(len(troubled_vars), 10)
                vars_to_flip = random.sample(troubled_vars, num_flips)

                for j in vars_to_flip:
                    if self.instance.var_types[j] == 'B': x_round[j] = 1 - x_round[j]
                # print(f"    Flipped {num_flips} variables.")

            # Step 2: Solve the projection LP
            proj_model = gp.Model("fp_projection")
            proj_model.Params.OutputFlag = 0
            proj_model.Params.Presolve = 0  # Disable presolve for stability

            x = proj_model.addVars(N, lb=self.instance.lb, ub=self.instance.ub, vtype=gp.GRB.CONTINUOUS, name="x")
            d = proj_model.addVars(I, lb=0.0, name="d")

            for i in range(self.instance.num_constraints):
                expr = gp.quicksum(self.instance.A[i, j] * x[j] for j in N if self.instance.A[i, j] != 0)
                sense = self.instance.sense[i]
                rhs = self.instance.b[i]
                if sense == 'L':
                    proj_model.addConstr(expr <= rhs)
                elif sense == 'E':
                    proj_model.addConstr(expr == rhs)

            for j in I:
                proj_model.addConstr(x[j] - x_round[j] <= d[j])
                proj_model.addConstr(x_round[j] - x[j] <= d[j])

            proj_model.setObjective(gp.quicksum(d[j] for j in I), GRB.MINIMIZE)
            proj_model.optimize()

            if proj_model.Status != GRB.OPTIMAL:
                # print(f"\n❌ Projection LP failed with status: {proj_model.Status}. Stopping pump.")
                return None, None

            distance = proj_model.ObjVal
            new_x_bar = np.array([proj_model.getVarByName(f"x[{i}]").X for i in N])
            new_int_infeas = self._count_integer_infeasibilities(new_x_bar)
            #
            # print(f"  Projection LP Solved:")
            # print(f"    Distance to Feasibility (LP Obj): {distance:.6f}")
            # print(f"    New Solution Integer Infeasibilities: {new_int_infeas}")

            if distance < 1e-6:
                # print("\n✅ SUCCESS! Feasibility Pump found a feasible solution.")
                sol = x_round  # If distance is 0, the rounded solution is feasible
                original_obj_val = sum(self.instance.obj[i] * sol[i] for i in N)
                # print(f"    Solution Objective: {original_obj_val:.5f}")
                # print("=======================================================")
                return original_obj_val,sol

            x_bar = new_x_bar
            previous_distance = distance
        #
        # print("\n❌ Feasibility Pump did not find a feasible solution after max iterations.")
        # print("=======================================================")
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
