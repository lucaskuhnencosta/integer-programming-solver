class LPSolver:
    def __init__(self, instance):
        self.template_model = instance.model

    def solve_relaxation(self, node):
        model = self.template_model.copy()
        for var, (lb, ub) in node.bounds.items():
            model.variables[var].lb = lb
            model.variables[var].ub = ub

        model.setParam("OutputFlag", 0)
        model.relax()
        model.optimize()

        if model.status != 2:  # Infeasible
            return None

        return model
