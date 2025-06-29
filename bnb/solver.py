import time

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


        gap_number=None
        best_bound = -float('inf')
        previous_best_bound = best_bound
        branching_times = []
        incumbent_str="     -     "
        gap_str = "  -  "



        # Gurobi-style column header
        print("    Nodes     |          Current Node         |        Objective Bounds        |  Time   |")
        print(" Expl  Unexpl |  Obj         Depth     IntInf | Incumbent    BestBd      Gap   |   (s)   |")
        print("--------------+-------------------------------+--------------------------------+----------+")
        mode="best-first"
        while not tree.empty():
            best_bound = tree.get_best_bound()

            if mode == "best-first":
                node=tree.pop()
            elif mode == "dfs":
                node=plunge_candidate

            active_mgr.switch_focus(node, working_model)
            start_time=time.time()
            node.evaluate_lp(working_model,self.instance)
            elapsed_time=time.time() - start_time

            if (self.enable_pump and node_counter%self.n_pump==0 and mode=="best-first"):
                self.feasibility_pump(node.solution,working_model)


            unexplored = len(tree.queue)

            if node.is_infeasible:
                self.prune_if_leaf(node)
                node_counter += 1
                continue

            if node.bound >= self.best_obj:
                self.prune_if_leaf(node)
                node_counter += 1
                mode="best-first"
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
                    gap_number=(incumbent - best_bound) / abs(incumbent)
                    gap_str = f"{gap_number*100:4.1f}%"
                    print(f"P{node_counter:5d} {unexplored:5d}  |{node.bound:8.5f}  {node.depth:5d}      {int_infeas:6d}  |{incumbent_str}  {best_bound:8.5f}  {gap_str}  |  {elapsed_time:4.2f}s")
                    if incumbent and gap_number and gap_number<self.gap_treshold:
                        break
                self.prune_if_leaf(node)
                node_counter += 1
                mode="best-first"
                continue

            if incumbent:
                gap_number = (incumbent - best_bound) / abs(incumbent)
                gap_str = f"{gap_number * 100:4.1f}%"

            if best_bound != previous_best_bound:
                print(f"D{node_counter:5d} {unexplored:5d}  |   -      {node.depth:5d}      {int_infeas:6d}  |{incumbent_str}  {best_bound:8.5f}  {gap_str}  |  {elapsed_time:4.2f}s")
            previous_best_bound=best_bound

            # Select branching variable using strong branching (or other strategy)

            branch_start = time.time()
            branch_var, left_node, right_node = self.brancher.select_branching_variable(node,node.solution,working_model,active_mgr)
            branch_end = time.time()
            branching_times.append(branch_end - branch_start)
            if branch_var is None or left_node is None or right_node is None:
                self.prune_if_leaf(node)
                node_counter += 1
                continue # No valid branching var found
            node_counter += 1
            node.children.extend([left_node, right_node])
            node.node_type = 'fork' if node.lp_basis else 'junction'

            if self.enable_plunging and (node_counter % self.k_plunging ==0):
                mode="dfs"

            if mode == "dfs":
                plunge_candidate=min(
                    [n for n in node.children if n.active and not n.is_infeasible],
                    key=lambda n:n.bound,
                default=None
                )
                if plunge_candidate:
                    if plunge_candidate==right_node and not left_node.is_infeasible:
                        left_node.type = "child"
                        tree.push(left_node)
                    elif plunge_candidate==left_node and not right_node.is_infeasible:
                        right_node.type = "child"
                        tree.push(right_node)
                    mode="dfs"
                    continue
                else:
                    mode="best-first"
                    continue

            if not left_node.is_infeasible:
                left_node.type = "child"
                tree.push(left_node)
            if not right_node.is_infeasible:
                right_node.type = "child"
                tree.push(right_node)

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


    def feasibility_pump(self,start_x,working_model):
        print("\n🔁 Starting Feasibility Pump...")

        I = [i for i, t in enumerate(self.instance.var_types) if t in ['B', 'I']]
        N = range(self.instance.num_vars)

        x_bar = start_x.copy()

        for iteration in range(self.fp_max_it):
            # Step 1: Round integer variables
            x_round = x_bar[:]
            for j in I:
                val=x_bar[j]
                lb, ub = self.instance.lb[j], self.instance.ub[j]

                lock_down=0
                lock_up=0

                A = self.instance.A
                lock_down = np.zeros(self.instance.num_vars, dtype=int)
                lock_up = np.zeros(self.instance.num_vars, dtype=int)

                for i in range(self.instance.num_constraints):
                    for j in range(self.instance.num_vars):
                        coeff = A[i, j]
                        if coeff > 0:
                            lock_up[j] += 1
                        elif coeff < 0:
                            lock_down[j] += 1

                for j in range(self.instance.num_vars):
                    if self.instance.var_types[j] in ['B', 'I']:
                        if lock_down[j] == 0:
                            x_round[j] = np.ceil(start_x[j])
                        elif lock_up[j] == 0:
                            x_round[j] = np.floor(start_x[j])
                        else:
                            x_round[j] = np.floor(start_x[j] + 0.5)
                    else:
                        x_round[j] = start_x[j]  # keep continuous vars as is

            # Step 2: Check if rounded solution is feasible
            working_model.update()
            for i, var in enumerate(working_model.getVars()):
                var.Start = x_round[i]
            working_model.params.StartNodeLimit = 1  # try solution directly
            working_model.optimize()
            if working_model.Status == GRB.OPTIMAL and self.instance.is_integral(
                    [v.X for v in working_model.getVars()]):
                obj_val = working_model.ObjVal
                print(f"✅ Feasibility pump found feasible solution at iteration {iteration}, obj = {obj_val:.5f}")
                if obj_val < self.best_obj:
                    self.best_obj = obj_val
                    self.best_sol = [v.X for v in working_model.getVars()]
                return self.best_obj, self.best_sol

            # Step 3: Solve projection LP: minimize sum |x_j - x̄_j| for j in I
            model = gp.Model("fp_projection")
            model.Params.OutputFlag = 0

            x = model.addVars(N, lb=self.instance.lb, ub=self.instance.ub, name="x")
            d = model.addVars(I, lb=0.0, name="d")

            # Add original Ax ≤ b constraints
            A = self.instance.A
            b = self.instance.b
            for i in range(self.instance.num_constraints):
                expr = gp.LinExpr()
                for j in range(self.instance.num_vars):
                    coeff = A[i, j]
                    if abs(coeff) > 1e-10:  # Skip zero coefficients
                        expr += coeff * x[j]
                model.addConstr(expr <= b[i])

            # Proximity constraints
            for j in I:
                model.addConstr(x[j] - x_round[j] <= d[j])
                model.addConstr(x_round[j] - x[j] <= d[j])

            # Objective: Minimize proximity
            model.setObjective(gp.quicksum(d[j] for j in I), GRB.MINIMIZE)

            model.optimize()
            if model.Status != GRB.OPTIMAL:
                print("❌ Feasibility pump projection LP was infeasible.")
                return

            x_bar = [x[i].X for i in N]

        print("❌ Feasibility pump did not find feasible solution after max iterations.")


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
