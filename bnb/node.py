import gurobipy as gp
class Node:
    def __init__(self, parent=None,depth=0, bound_changes=None,node_number=0):
        self.parent = parent
        self.bound_changes = bound_changes or {}  # e.g., {i: (lb, ub)}
        self.depth = depth
        self.node_number = node_number

        # BnB state
        self.bound=None #LP objective value
        self.solution = None #List of variable values (solutions of relaxation)
        self.status=None #Gurobi solver status
        self.is_infeasible = False
        self.is_integer = False

        # Tree structure
        self.children = []
        self.active = True
        self.node_type='unprocessed' # One of: focusnode, child, sibling, leaf, fork, junction
        self.fork_parent=None

        # Basis storage
        self.lp_basis=None

    def accumulated_bounds(self):
        bounds = {}
        node=self
        while node:
            for var, (lb,ub) in node.bound_changes.items():
                if var not in bounds:
                    bounds[var] = (lb,ub)
            node=node.parent
        return bounds

    def evaluate_lp(self,model,instance):
        model.optimize()
        self.status=model.Status
        if model.Status == gp.GRB.Status.INFEASIBLE:
            self.is_infeasible = True
            self.bound=float("inf")
            return
        self.bound=model.ObjVal
        self.solution=[var.X for var in model.getVars()]
        self.is_integer=all(
            abs(self.solution[i]-round(self.solution[i])) <1e-6
            for i, vtype in enumerate(instance.var_types)
            if vtype in ['B','I']
        )
        try:
            v_basis=model.getAttr("VBasis")
            c_basis=model.getAttr("CBasis")
            self.lp_basis=(v_basis,c_basis)
        except gp.GurobiError:
            self.lp_basis=None

        self.fork_parent=self if self.lp_basis else (self.parent.fork_parent if self.parent else None)