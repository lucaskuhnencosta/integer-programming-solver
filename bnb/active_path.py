import gurobipy as gp

class ActivePathManager:
    def __init__(self, root_node,instance):
        self.active_path = [root_node]  # from root to current focus node
        self.focus = root_node
        self.instance = instance

    def find_common_ancestor(self, node):
        """Finds the deepest common ancestor between current focus and node."""
        focus_ancestors = set()
        current = self.focus
        while current:
            focus_ancestors.add(current)
            current = current.parent

        while node not in focus_ancestors:
            node = node.parent
        return node

    def switch_focus(self, new_node, model):
        """
        Switch the current LP model from focus → new_node by:
        - Undoing bounds up to the common ancestor.
        - Reapplying bounds down to the new focus node.
        """
        ancestor = self.find_common_ancestor(new_node)

        # Step 1: Undo changes up to ancestor
        current = self.focus
        while current != ancestor:
            self._undo_changes(current, model)
            current = current.parent

        # Step 2: Build path down to new node
        path_down = []
        node = new_node
        while node != ancestor:
            path_down.append(node)
            node = node.parent
        path_down.reverse()

        # Step 3: Reapply changes
        for node in path_down:
            self._apply_changes(node, model)

        # Step 4: Load LP warm start from fork parent if it exists
        if new_node.fork_parent and new_node.fork_parent.lp_basis:
            v_basis, c_basis = new_node.fork_parent.lp_basis
            try:
                vars = model.getVars()
                constrs = model.getConstrs()
                for var, vb in zip(vars, v_basis):
                    var.VBasis = vb
                for constr, cb in zip(constrs, c_basis):
                    constr.CBasis = cb
            except gp.GurobiError:
                print("CS")

        self.focus = new_node
        self.active_path = self._rebuild_path(new_node)

    def _undo_changes(self, node, model):
        for var_idx, _ in node.bound_changes.items():
            var = model.getVars()[var_idx]
            var.lb = -self.instance.lb[var_idx]
            var.ub = self.instance.ub[var_idx]

    def _apply_changes(self, node, model):
        for var_idx, (lb, ub) in node.bound_changes.items():
            var = model.getVars()[var_idx]
            var.lb = lb
            var.ub = ub

    def _rebuild_path(self, node):
        path = []
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))
