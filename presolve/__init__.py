# def feasibility_pump(self, start_x, working_model):
#     print("\n🔁 Starting Feasibility Pump...")
#
#     I = [i for i, t in enumerate(self.instance.var_types) if t in ['B', 'I']]
#     N = range(self.instance.num_vars)
#     x_bar = np.array(start_x)
#
#     for iteration in range(self.fp_max_it):
#         print("iteration:", iteration)
#         # Step 1: Round integer variables
#         x_round = x_bar.copy()
#         for j in I:
#             x_round[j] = round(x_bar[j])
#
#         # Step 2: Solve the projection LP to find a feasible 'x' closest to 'x_round'
#         proj_model = gp.Model("fp_projection")
#         proj_model.Params.OutputFlag = 0
#
#         # Define variables
#         x = proj_model.addVars(N, lb=self.instance.lb, ub=self.instance.ub, vtype=gp.GRB.CONTINUOUS, name="x")
#         d = proj_model.addVars(I, lb=0.0, name="d")  # Proximity variables for L1-norm
#
#         # Add original Ax<=b constraints
#         for i in range(self.instance.num_constraints):
#             expr = gp.quicksum(self.instance.A[i, j] * x[j] for j in N if self.instance.A[i, j] != 0)
#             sense = self.instance.sense[i]
#             rhs = self.instance.b[i]
#
#             if sense == 'L':
#                 proj_model.addConstr(expr <= rhs)
#             elif sense == 'E':
#                 proj_model.addConstr(expr == rhs)
#
#         # Proximity constraints to model the L1-norm
#         for j in I:
#             proj_model.addConstr(x[j] - x_round[j] <= d[j])
#             proj_model.addConstr(x_round[j] - x[j] <= d[j])
#
#         # Objective: Minimize the L1-distance to the rounded solution
#         proj_model.setObjective(gp.quicksum(d[j] for j in I), GRB.MINIMIZE)
#         proj_model.optimize()
#
#         if proj_model.Status == GRB.INFEASIBLE:
#             print("❌ Feasibility Pump projection LP was infeasible. Stopping pump.")
#             return
#
#             # If the distance is zero, we have found a feasible integer solution
#         if proj_model.ObjVal < 1e-6:
#             sol = [x[i].X for i in N]
#             original_obj_val = sum(self.instance.obj[i] * sol[i] for i in N)
#
#             print(f"✅ Feasibility Pump found a feasible solution with obj = {original_obj_val:.5f}")
#             if original_obj_val < self.best_obj:
#                 print("    (New incumbent found!)")
#                 self.best_obj = original_obj_val
#                 self.best_sol = sol
#             return
#
#             # Update x_bar with the new LP-feasible point for the next iteration
#         x_bar = np.array([x[i].X for i in N])
#
#         print("❌ Feasibility Pump did not find a feasible solution after max iterations.")