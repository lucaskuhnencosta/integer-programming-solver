def feasibility_pump(self, start_x, working_model):
    print("\n🔁 Starting Feasibility Pump...")

    I = [i for i, t in enumerate(self.instance.var_types) if t in ['B', 'I']]
    N = range(self.instance.num_vars)
    x_bar = np.array(start_x)

    for iteration in range(self.fp_max_it):
        # Step 1: Round integer variables
        x_round = x_bar.copy()
        for j in I:
            x_round[j] = round(x_bar[j])
            # val=x_bar[j]
            # lb, ub = self.instance.lb[j], self.instance.ub[j]

            # lock_down=0
            # lock_up=0
            #
            # A = self.instance.A
            # lock_down = np.zeros(self.instance.num_vars, dtype=int)
            # lock_up = np.zeros(self.instance.num_vars, dtype=int)
            #
            # for i in range(self.instance.num_constraints):
            #     for j in range(self.instance.num_vars):
            #         coeff = A[i, j]
            #         if coeff > 0:
            #             lock_up[j] += 1
            #         elif coeff < 0:
            #             lock_down[j] += 1

            # for j in range(self.instance.num_vars):
            #     if self.instance.var_types[j] in ['B', 'I']:
            #         if lock_down[j] == 0:
            #             x_round[j] = np.ceil(start_x[j])
            #         elif lock_up[j] == 0:
            #             x_round[j] = np.floor(start_x[j])
            #         else:
            #             x_round[j] = np.floor(start_x[j] + 0.5)
            #     else:
            #         x_round[j] = start_x[j]  # keep continuous vars as is

        # Step 2: Solve the projection LP to find a feasible 'x' closest to 'x_round'
        proj_model = gp.Model("fp_projection")
        proj_model.Params.OutputFlag = 0

        # Define variables
        x = proj_model.addVars(N, lb=self.instance.lb, ub=self.instance.ub, vtype=gp.GRB.CONTINUOUS, name="x")
        d = proj_model.addVars(I, lb=0.0, name="d")  # Proximity variables for L1-norm

        # working_model.update()
        # for i, var in enumerate(working_model.getVars()):
        #     var.Start = x_round[i]
        # working_model.params.StartNodeLimit = 1  # try solution directly
        # working_model.optimize()
        # if working_model.Status == GRB.OPTIMAL and self.instance.is_integral(
        #         [v.X for v in working_model.getVars()]):
        #     obj_val = working_model.ObjVal
        #     print(f"✅ Feasibility pump found feasible solution at iteration {iteration}, obj = {obj_val:.5f}")
        #     if obj_val < self.best_obj:
        #         self.best_obj = obj_val
        #         self.best_sol = [v.X for v in working_model.getVars()]
        #     return self.best_obj, self.best_sol

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
    node.active = False
    parent = node.parent
    while parent:
        parent.children = [c for c in parent.children if c.active]
        if not parent.children:
            parent.active = False
            parent = parent.parent
        else:
            break













          # if mode == "dfs":
                #     plunge_candidate=min(
                #         [n for n in node.children if n.active and not n.is_infeasible],
                #         key=lambda n:n.bound,
                #     default=None
                #     )
                #     if plunge_candidate:
                #         if plunge_candidate==right_node and not left_node.is_infeasible:
                #             left_node.type = "child"
                #             tree.push(left_node)
                #         elif plunge_candidate==left_node and not right_node.is_infeasible:
                #             right_node.type = "child"
                #             tree.push(right_node)
                #         mode="dfs"
                #         continue
                #     else:
                #         mode="best-first"
                #         continue
