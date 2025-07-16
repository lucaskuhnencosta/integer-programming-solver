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
    def __init__(self, mip_instance,enable_plunging=False,k_plunging=10,clique_cuts=False,strong_depth=10,strong_k=1500):

        self.instance = mip_instance

        ### Branching ###
        self.strong_depth = strong_depth
        self.strong_k = strong_k
        self.brancher = Branching(self.instance,strong_depth=self.strong_depth, k=self.strong_k)

        ### Clique cuts ###
        self.enable_clique_cuts = clique_cuts

        ### Plunging
        self.enable_plunging = enable_plunging
        self.k_plunging = k_plunging

        ### Primal Heuristics ###
        self.enable_pump=True
        self.n_pump = 2
        self.fp_max_it=20000
        self.pump_model = None
        self.pump_x_vars = None
        self.pump_d_vars = None
        self.pump_dist_constrs_le = None
        self.pump_dist_constrs_ge = None

        self.enable_diving=True
        self.n_diving=7

        ### Other primal heuristics parameters ###


        self._compute_locks()

        self.gap_treshold = 1e-3 #%


        self.best_obj = float("inf")
        self.best_sol=None

        self.max_processed_depth=0
        self.plunge_steps_done=0


        self.I = [i for i, t in enumerate(self.instance.var_types) if t in ['B', 'I']]
        self.B = [i for i, t in enumerate(self.instance.var_types) if t in ['B']]


    def solve(self):
        ### Initialization of variables
        start_total_solver_time = time.time()
        times=[]
        primal_bounds=[]
        dual_bounds=[]
        TIMEOUT_SECONDS = 300

        ### Initializing cliques

        if self.enable_clique_cuts:
            start_cliques_time = time.time()
            self.strengthened_cliques = []
            self.instance._complement_all_binary_vars()
            print("⚙️  Pre-computing cliques for cut generation...")
            binary_A, binary_b = self.instance.get_binary_subproblem()
            print(f"    Found {len(binary_A)} binary-only constraints for analysis.")
            seed_cliques = clique_detection(binary_A, binary_b)
            print(f"    Detected {len(seed_cliques)} seed cliques.")
            conflict_graph = graph_builder(seed_cliques)
            print(
                f"    Built conflict graph with {conflict_graph.number_of_nodes()} nodes and {conflict_graph.number_of_edges()} edges.")
            self.strengthened_cliques = clique_extension(conflict_graph, seed_cliques)
            elapsed_cliques_time=time.time()-start_cliques_time
            print(f"    Extended to {len(self.strengthened_cliques)} maximal cliques in {elapsed_cliques_time}.")

        ### Assembling a model for the root

        self.instance.build_root_model()
        working_model = self.instance.root_lp_model.copy()
        working_model.setParam("OutputFlag", 0)

        ### Creating the root as a node

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
            for i in range(20):
                num_new_cuts=self._separate_clique_cuts(root,working_model)
                if num_new_cuts>0:
                    root.evaluate_lp(working_model, self.instance)
                else:
                    break
        print("Root relaxation: objective %.8f, time %.2f seconds\n" % (root.bound, elapsed_time))

        ### Creating tree and activity manager

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


        incumbent_str="     -     "
        gap_str = "  -  "
        print("    Nodes     |          Current Node         |        Objective Bounds        |  Time   |")
        print(" Expl  Unexpl |  Obj         Depth     IntInf | Incumbent    BestBd      Gap   |   (s)   |")
        print("--------------+-------------------------------+--------------------------------+----------+")
        try:
            start_tree_time=time.time()
            while not tree.empty():

                ########## CHECK IF WE HAVE TIME #########

                if time.time() - start_total_solver_time > TIMEOUT_SECONDS:
                    print("\n⏰ Timeout limit reached. Terminating search.")
                    break  # Exit the loop gracefully

                ########## NODE SELECTION #########

                current_tree_best_bound = tree.get_best_bound()
                if mode == "dfs":
                    node=tree.pop_dfs()
                    if node is None:
                        mode="best-first"
                        continue
                else:
                    node=tree.pop_best_bound()
                    if node is None:
                        break

                ########## NODE SOLVE #########

                node_counter += 1
                active_mgr.switch_focus(node, working_model)
                node.evaluate_lp(working_model,self.instance)
                if self.enable_clique_cuts and node.depth % 5 ==0:
                    num_new_cuts = self._separate_clique_cuts(node, working_model)
                    if num_new_cuts > 0:
                        node.evaluate_lp(working_model, self.instance)
                elapsed_time = time.time() - start_tree_time

                ########## NODE PRUNNING #########

                unexplored = len(tree)
                if node.is_infeasible or node.bound>self.best_obj:
                    self.prune_if_leaf(node)
                    continue

                ########## STATS AND INFO #########

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
                        gap_number=(incumbent - current_tree_best_bound) / abs(incumbent)
                        gap_str = f"{gap_number*100:4.1f}%"
                        print(f"P{node_counter:5d} {unexplored:5d}  |{node.bound:8.5f}  {node.depth:5d}      {int_infeas:6d}  |{incumbent_str}  {current_tree_best_bound:8.5f}  {gap_str}  |  {elapsed_time:4.2f}s")
                        times.append(elapsed_time)
                        primal_bounds.append(incumbent)
                        dual_bounds.append(current_tree_best_bound)

                        if incumbent and gap_number and gap_number<self.gap_treshold:
                            break
                    self.prune_if_leaf(node)
                    continue
                if incumbent:
                    primal_bounds.append(incumbent)
                    gap_number = (incumbent - current_tree_best_bound) / abs(incumbent)
                    gap_str = f"{gap_number * 100:4.1f}%"
                else:
                    primal_bounds.append(None)
                print(f"D{node_counter:5d} {unexplored:5d}  |   -      {node.depth:5d}      {int_infeas:6d}  |{incumbent_str}  {current_tree_best_bound:8.5f}  {gap_str}  |  {elapsed_time:4.2f}s")
                times.append(elapsed_time)
                dual_bounds.append(current_tree_best_bound)

                ########## PRIMAL HEURISTICS ZONE #########

                if not incumbent and self.enable_pump and (node_counter%self.n_pump==0):
                    if node.solution is not None:
                        fp_obj,fp_sol=self.feasibility_pump(node.solution, working_model)
                        if fp_obj is not None:
                            self.best_obj = fp_obj
                            self.best_sol = fp_sol
                            incumbent = self.best_obj
                            incumbent_str = f"{incumbent:10.5f}"
                            print(f"FP FOUND NEW INCUMBENT: {self.best_obj:.5f}")

                if self.enable_diving and (node_counter%self.n_diving==0):
                    if node.solution is not None:
                        dv_obj,dv_sol=self._diving_heuristic(node, working_model,active_mgr)
                        if dv_obj is not None and dv_obj<self.best_obj:
                            self.best_obj = dv_obj
                            self.best_sol = dv_sol
                            incumbent = self.best_obj
                            incumbent_str = f"{incumbent:10.5f}"
                            print(f"DV FOUND NEW INCUMBENT: {self.best_obj:.5f}")

                ########## BRANCHING ZONE #########

                branch_var, left_node, right_node = self.brancher.select_branching_variable(node,node.solution,working_model,active_mgr,clique_cuts=self.enable_clique_cuts)
                if branch_var is None:
                    self.prune_if_leaf(node)
                    continue
                node.children.extend([left_node, right_node])
                node.node_type = 'fork' if node.lp_basis else 'junction'
                tree.push_children(left_node,right_node)

                ########## NODE SELECTION ZONE #########

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
                    if incumbent is not None and current_tree_best_bound > -float('inf'):
                        denominator=self.best_obj - current_tree_best_bound
                        if denominator>1e-6:
                            gamma=(node.bound-current_tree_best_bound)/denominator
                            if gamma>0.25:
                                stop_plunging=True
                    if stop_plunging:
                        mode="best-first"
                        self.plunge_steps_done=0
                if self.enable_plunging and (node_counter>1 and node_counter % self.k_plunging ==0):
                    if mode!="dfs":
                        mode="dfs"

                ########## EXCEPTION ZONE #########

        except Exception as e:
            import traceback
            traceback.print_exc()
        except KeyboardInterrupt:
            print("\n\n🛑 User interrupt detected! Stopping search.")
            print("Returning best solution found so far.")

        ########## END ZONE #########

        finally:
            end_total_solver_time = time.time()
            elapsed_total_solver_time = end_total_solver_time - start_total_solver_time
            print("Total solver time: {:.4f} seconds".format(elapsed_total_solver_time))
            print(f"Total number of nodes explored:{node_counter}")

            return self.best_sol, self.best_obj, times, primal_bounds, dual_bounds



    def _build_pump_model(self):
        N = range(self.instance.num_vars)
        self.pump_model = gp.Model("fp_projection_reusable")
        self.pump_model.Params.OutputFlag = 0
        self.pump_x_vars = self.pump_model.addVars(N, lb=self.instance.lb, ub=self.instance.ub, name="x")
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
                        final_obj = np.dot(self.instance.obj, x_bar) #+self.instance.obj_const
                        return final_obj, x_bar

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, None

        return None, None



    def _separate_clique_cuts(self,node,working_model):
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
            working_model.update()  # Apply changes to the model
        return cuts_added



    def _compute_locks(self):
        num_vars = self.instance.num_vars
        self.down_locks = np.zeros(num_vars)
        self.up_locks = np.zeros(num_vars)
        for i in range(self.instance.num_constraints):
            sense = self.instance.sense[i]
            row = self.instance.A[i, :]
            for j in np.nonzero(row)[0]:
                coeff = row[j]
                if sense == 'L':
                    if coeff > 0:
                        self.down_locks[j] += 1
                    else:
                        self.up_locks[j] += 1
                elif sense == 'E':
                    self.down_locks[j] += 1
                    self.up_locks[j] += 1



    def _simple_rounding_heuristic(self, lp_solution,fractional_vars):
        rounded_solution = lp_solution.copy()
        for var_idx,_ in fractional_vars:
            if self.down_locks[var_idx] == 0:
                rounded_solution[var_idx] = np.floor(lp_solution[var_idx])
            elif self.up_locks[var_idx] == 0:
                rounded_solution[var_idx] = np.ceil(lp_solution[var_idx])
        return rounded_solution



    def _diving_heuristic(self, node, working_model, active_path, max_dive_depth=20000):
        original_focus = active_path.focus
        current_solution = node.solution
        current_node = node
        for dive_step in range(max_dive_depth):
            fractional_vars = [(j, current_solution[j]) for j in self.I if abs(current_solution[j] - round(current_solution[j])) > 1e-6]
            if not fractional_vars:
                dive_obj = np.dot(self.instance.obj, current_solution) #+ self.instance.obj_const
                return dive_obj,current_solution

            sr_sol = self._simple_rounding_heuristic(current_solution,fractional_vars)
            fractional_vars = [(j, sr_sol[j]) for j in self.I if abs(sr_sol[j] - round(sr_sol[j])) > 1e-6]
            if not fractional_vars:
                dive_obj = np.dot(self.instance.obj, sr_sol)# + self.instance.obj_const
                return dive_obj, sr_sol

            binary_fractional_vars_after_round = [(j, sr_sol[j]) for j in self.B if abs(sr_sol[j] - round(sr_sol[j])) > 1e-6]

            if not binary_fractional_vars_after_round:
                fractional_vars_after_round = fractional_vars
            else:
                fractional_vars_after_round = binary_fractional_vars_after_round

            best_dive_var = -1
            min_lock_score = float('inf')
            min_dist_to_int = float('inf')  # Tie-breaker: distance to nearest integer
            for var_idx, frac_value in fractional_vars_after_round:
                down_lock=self.down_locks[var_idx]
                up_lock=self.up_locks[var_idx]

                current_score = min(down_lock, up_lock)

                if current_score < min_lock_score:
                    min_lock_score = current_score
                    best_dive_var = var_idx
                    min_dist_to_int = min(frac_value - np.floor(frac_value), np.ceil(frac_value) - frac_value)

                elif current_score==min_lock_score:
                    current_dist_to_int = min(frac_value - np.floor(frac_value), np.ceil(frac_value) - frac_value)
                    if current_dist_to_int < min_dist_to_int:
                        best_dive_var = var_idx
                        min_dist_to_int = current_dist_to_int

            if best_dive_var == -1:
                best_dive_var = max(fractional_vars_after_round, key=lambda item: 0.5 - abs(item[1] - 0.5))[0]

            dive_down = self.down_locks[best_dive_var] <= self.up_locks[best_dive_var]
            val = current_solution[best_dive_var]
            bound_change = {}
            if dive_down:
                # print(f"   [Dive Step {dive_step + 1}] Diving DOWN on var {best_dive_var} (value {val:.2f})")
                bound_change = {best_dive_var: (self.instance.lb[best_dive_var], np.floor(val))}
            else:
                # print(f"   [Dive Step {dive_step + 1}] Diving UP on var {best_dive_var} (value {val:.2f})")
                bound_change = {best_dive_var: (np.ceil(val), self.instance.ub[best_dive_var])}
            dive_node = Node(parent=current_node, bound_changes=bound_change, depth=current_node.depth + 1)
            active_path.switch_focus(dive_node, working_model)
            dive_node.evaluate_lp(working_model, self.instance)
            if dive_node.is_infeasible:
                break

            current_solution = dive_node.solution
            current_node = dive_node

        active_path.switch_focus(original_focus, working_model)
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
