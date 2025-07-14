import time
import random
import numpy as np

from bnb.active_path import ActivePathManager
from bnb.node import Node
from bnb.branching import Branching
# from bnb.shared_state import SharedState
from bnb.tree import BranchAndBoundTree

from cutgen.Cliques import *
from cutgen.graph_builder import *

import gurobipy as gp
from gurobipy import GRB

class BranchAndBoundSolver:
    def __init__(self, mip_instance,enable_plunging=False,k_plunging=10,enable_pump=False,n_pump=100,fp_max_it=1000,clique_cuts=False):
        self.instance = mip_instance

        # self.shared = SharedState()

        self.brancher = Branching(self.instance)
        self.gap_treshold = 1e-3 #%

        self.enable_plunging = enable_plunging
        self.k_plunging = k_plunging

        self.enable_pump = enable_pump
        self.n_pump = n_pump
        self.fp_max_it=fp_max_it

        self.best_obj = float("inf")
        self.best_sol=None

        self.max_processed_depth=0
        self.plunge_steps_done=0

        self.pump_model = None
        self.pump_x_vars = None
        self.pump_d_vars = None
        self.pump_dist_constrs_le = None
        self.pump_dist_constrs_ge = None
        self.I = [i for i, t in enumerate(self.instance.var_types) if t in ['B', 'I']]
        self.enable_clique_cuts=clique_cuts

    def solve(self):
        start_total_solver_time = time.time()


        if self.enable_clique_cuts:
            self.strengthened_cliques = []

            self.instance._complement_all_binary_vars()
            print("⚙️  Pre-computing cliques for cut generation...")

            # 1a. Extract the binary-only constraints from the instance
            binary_A, binary_b = self.instance.get_binary_subproblem()
            print(f"    Found {len(binary_A)} binary-only constraints for analysis.")

            # 1b. Detect initial "seed" cliques from these constraints
            seed_cliques = clique_detection(binary_A, binary_b)
            print(f"    Detected {len(seed_cliques)} seed cliques.")

            # 1c. Build the conflict graph from the cliques
            conflict_graph = graph_builder(seed_cliques)
            print(
                f"    Built conflict graph with {conflict_graph.number_of_nodes()} nodes and {conflict_graph.number_of_edges()} edges.")

            # 1d. Extend the cliques to make them stronger
            self.strengthened_cliques = clique_extension(conflict_graph, seed_cliques)
            print(f"    Extended to {len(self.strengthened_cliques)} maximal cliques.")

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

        if self.enable_clique_cuts:
            for i in range(10):
                num_new_cuts=self._separate_clique_cuts(root,working_model)
                if num_new_cuts>0:
                    root.evaluate_lp(working_model, self.instance)
                else:
                    break

        print("\nRoot relaxation: objective %.8f, time %.2f seconds\n" % (root.bound, elapsed_time))

        tree = BranchAndBoundTree()
        active_mgr = ActivePathManager(root,self.instance)

        branch_var, left_node, right_node = self.brancher.select_branching_variable(root, root.solution, working_model,active_mgr,clique_cuts=self.enable_clique_cuts)

        if branch_var is None:
            print("✅ Root node is optimal or infeasible after cuts.")
            # ... handle this case ...
            return root.solution, root.bound + self.instance.obj_const

        tree.push_children(left_node, right_node)

        node_counter = 0
        incumbent = None
        print_stats=False

        mode = "best-first"
        self.max_processed_depth=0
        self.plunge_steps_done=0

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
                # print("We are here")
                current_tree_best_bound = tree.get_best_bound()

                if mode == "dfs":
                    node=tree.pop_dfs()
                    # print("We are here in dfs mode")
                    if node is None:
                        mode="best-first"
                        # print("The node is None")
                        continue

                else:
                    node=tree.pop_best_bound()
                    # print("We are here in best first mode popping a node")
                    if node is None:
                        # print("The node is None")
                        break

                node_counter += 1
                active_mgr.switch_focus(node, working_model)
                start_time=time.time()
                node.evaluate_lp(working_model,self.instance)
                elapsed_time=time.time() - start_time

                if self.enable_clique_cuts and node.depth % 5 ==0:
                    num_new_cuts = self._separate_clique_cuts(node, working_model)
                    # If we added cuts, we must re-solve the LP to get a new, tighter bound
                    if num_new_cuts > 0:
                        node.evaluate_lp(working_model, self.instance)

                unexplored = len(tree)

                if node.is_infeasible:
                    self.prune_if_leaf(node)
                    continue


                if node.bound>=self.best_obj:
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
                    continue

                if not incumbent and self.enable_pump and (node_counter%self.n_pump==0):
                    if node.solution is not None:
                        fp_obj,fp_sol=self.feasibility_pump(node.solution, working_model)
                        if fp_obj is not None:
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
                branch_var, left_node, right_node = self.brancher.select_branching_variable(node,node.solution,working_model,active_mgr,clique_cuts=self.enable_clique_cuts)
                branch_end = time.time()
                branching_times.append(branch_end - branch_start)
                if branch_var is None:
                    self.prune_if_leaf(node)
                    continue # No valid branching var found

                node.children.extend([left_node, right_node])
                node.node_type = 'fork' if node.lp_basis else 'junction'

                tree.push_children(left_node,right_node)
                self.max_processed_depth = max(self.max_processed_depth, node.depth)

                if mode == "dfs":
                    self.plunge_steps_done+=1
                    stop_plunging=False
                    min_steps=0.1*self.max_processed_depth
                    max_steps=0.5*self.max_processed_depth
                    if self.plunge_steps_done < min_steps:
                        continue

                    if self.plunge_steps_done >= max_steps:
                        stop_plunging=True

                    if incumbent is not None and global_best_bound > -float('inf'):
                        denominator=self.best_obj - global_best_bound
                        if denominator>1e-6:
                            gamma=(node.bound-global_best_bound)/denominator
                            if gamma>0.25:
                                stop_plunging=True

                    if stop_plunging:
                        mode="best-first"
                        self.plunge_steps_done=0


                if self.enable_plunging and (node_counter>1 and node_counter % self.k_plunging ==0):
                    if mode!="dfs":
                        mode="dfs"





        except Exception as e:
            print("\n\n🚨 EXCEPTION CAUGHT AFTER ADDING CUTS 🚨")
            import traceback
            traceback.print_exc()
            # Re-raise the exception to stop the program
            raise e
        # --- END DEBUGGING BLOCK ---
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

    # In bnb/solver.py, as a new method in the BranchAndBoundSolver class
    def _build_pump_model(self):
        N = range(self.instance.num_vars)
        self.pump_model = gp.Model("fp_projection_reusable")
        self.pump_model.Params.OutputFlag = 0
        # The corrected line
        self.pump_x_vars = self.pump_model.addVars(N, lb=self.instance.lb, ub=self.instance.ub, name="x")
        # self.pump_x_vars = self.pump_model.addVars(N, lb=self.instance.lb.tolist(), ub=self.instance.ub.tolist(),name="x")
        self.pump_d_vars = self.pump_model.addVars(self.I, lb=0.0, name="d")
        for i in range(self.instance.num_constraints):
            expr = gp.quicksum(self.instance.A[i, j] * self.pump_x_vars[j] for j in N if self.instance.A[i, j] != 0)
            sense = self.instance.sense[i]
            rhs = self.instance.b[i]
            if sense == 'L':
                self.pump_model.addConstr(expr <= rhs)
            elif sense == 'E':
                self.pump_model.addConstr(expr == rhs)

        self.pump_dist_constrs_le = {j: self.pump_model.addConstr(self.pump_x_vars[j] - self.pump_d_vars[j] <= 0) for j in self.I}
        self.pump_dist_constrs_ge = {j: self.pump_model.addConstr(self.pump_x_vars[j] + self.pump_d_vars[j] >= 0) for j in self.I}
        self.pump_model.setObjective(gp.quicksum(self.pump_d_vars[j] for j in self.I), GRB.MINIMIZE)

    def feasibility_pump(self, start_x, working_model):
        if self.pump_model is None:
            self._build_pump_model()

        x_bar = np.array(start_x)
        previous_distance = float('inf')
        stall_counter = 0

        for iteration in range(self.fp_max_it):
            try:
                x_round = np.round(x_bar) # Simplified rounding

                if self._check_lp_feasibility(x_round):
                    final_obj = np.dot(self.instance.obj, x_round)
                    return final_obj, x_round

                if iteration > 0 and 'distance' in locals() and distance >= previous_distance:
                    stall_counter += 1
                    if stall_counter > 20:
                        return None, None
                    troubled_vars = [j for j in self.I if abs(x_bar[j] - x_round[j]) > 1e-6]
                    if not troubled_vars: troubled_vars = self.I
                    num_flips = min(len(troubled_vars), 10)
                    vars_to_flip = random.sample(troubled_vars, num_flips)
                    for j in vars_to_flip:
                        if self.instance.var_types[j] == 'B': x_round[j] = 1 - x_round[j]
                else:
                    stall_counter = 0

                for j in self.pump_dist_constrs_le:
                    self.pump_dist_constrs_le[j].RHS = x_round[j]
                    self.pump_dist_constrs_ge[j].RHS = x_round[j]

                self.pump_model.optimize()

                if self.pump_model.Status != GRB.OPTIMAL: return None, None

                distance = self.pump_model.ObjVal
                previous_distance = distance
                x_bar = np.array(list(self.pump_model.getAttr("X", self.pump_x_vars).values()))
                if distance < 1e-6:
                    if self._count_integer_infeasibilities(x_bar) == 0:
                        final_obj = np.dot(self.instance.obj, x_bar)
                        return final_obj, x_bar

            except Exception as e:
                print(f"\n\n🚨 EXCEPTION CAUGHT IN FEASIBILITY PUMP 🚨")
                import traceback
                traceback.print_exc()
                return None, None

        return None, None

    def _separate_clique_cuts(self,node,working_model):
        """
        Finds and adds violated clique cuts for the current node's LP solution.
        Returns the number of cuts added
        """

        if not self.strengthened_cliques or node.solution is None:
            return 0

        cuts_added = 0
        lp_solution = node.solution

        for clique in self.strengthened_cliques:
            if sum(lp_solution[j] for j in clique) > 1.0 + 1e-6:
                vars_in_cut=[working_model.getVarByName(self.instance.var_names[j]) for j in clique]
                working_model.addConstr(gp.quicksum(vars_in_cut) <= 1)

                cuts_added += 1
        if cuts_added >0:
            # print(f"    -> Added {cuts_added} clique cuts to the LP relaxation.")
            working_model.update()  # Apply changes to the model

        return cuts_added


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
