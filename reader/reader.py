import cplex
import numpy as np
import gurobipy as gp
from typing import List, Tuple, Dict


class MIPInstance:
    def __init__(self, mps_path: str):
        self.mps_path = mps_path

        self.model = cplex.Cplex()
        self.model.set_results_stream(None)
        self.model.set_warning_stream(None)
        self.model.set_error_stream(None)
        self.model.set_log_stream(None)
        self.model.read(mps_path)

        # Attributes to be populated
        # self.num_vars = 0
        # self.num_constraints = 0
        self.obj = []
        self.A = None  # Constraint matrix
        self.b = []
        self.sense = []
        self.lb = []
        self.ub = []
        self.var_types = []
        self.var_names = []

        self.root_lp_model=None


        # Extract all necessary data
        self._extract_data()

    @property
    def num_vars(self) -> int:
        return len(self.var_names)

    @property
    def num_constraints(self) -> int:
        return len(self.row_names)

    @property
    def num_binary(self) -> int:
        return self.var_types.count('B')

    @property
    def num_integer(self) -> int:
        return self.var_types.count('I')

    @property
    def num_continuous(self) -> int:
        return self.var_types.count('C')


    def _extract_data(self):
        ### Objective ###
        self.obj = self.model.objective.get_linear()

        ### Variables, their types, bounds, names, and number ###
        self.var_types = self.model.variables.get_types()
        self.lb = self.model.variables.get_lower_bounds()
        self.ub = self.model.variables.get_upper_bounds()
        self.var_names = self.model.variables.get_names()
        self._clean_variable_names()


        # Map CPLEX variable indices to our internal dense index
        self.var_index_map = {name: idx for idx, name in enumerate(self.var_names)}
        self.reverse_var_index = {v: k for k, v in self.var_index_map.items()}

        ### Constraints number###
        # self.num_constraints = self.model.linear_constraints.get_num()
        self.row_names = self.model.linear_constraints.get_names()
        lin_expr = self.model.linear_constraints.get_rows()
        self.A = self._build_constraint_matrix(lin_expr)
        self.sense = self.model.linear_constraints.get_senses()
        self.b = self.model.linear_constraints.get_rhs()





    def _build_constraint_matrix(self, lin_expr: List[cplex.SparsePair]) -> np.ndarray:
        A = np.zeros((self.num_constraints, self.num_vars))
        var_names=self.var_names

        for row_idx, row in enumerate(lin_expr):
            for cplex_idx, coeff in zip(row.ind, row.val):
                var_name=var_names[cplex_idx]
                dense_idx=self.var_index_map[var_name]
                A[row_idx][dense_idx] = coeff
        return A

    def pretty_print(self):
        print(f'This model has originally {self.num_vars} variables and {self.num_constraints} constraints')
        ##########################################################################
        print("\n=== Objective ===")
        print("Objective vector:", self.obj)

        ###########################################################################
        print('\n=== Variables ===')

        print(f"Total binary variables: {self.num_binary}")
        print(f"Total integer variables: {self.num_integer}")
        print(f"Total continuous variables: {self.num_continuous}")

        print("Variable types:", self.var_types)
        print("Variable names:", self.var_names)
        print("Bounds:")
        for name, lb, ub in zip(self.var_names, self.lb, self.ub):
            if abs(lb - ub) < 1e-8:
                print(f"  {name}: FIXED to {lb}")
            else:
                lb_str = str(lb) if lb > -1e20 else "-∞"
                ub_str = str(ub) if ub < 1e20 else "∞"
                print(f"  {name}: [{lb_str}, {ub_str}]")

        print("Variable Types:")
        for name, vtype in zip(self.var_names, self.var_types):
            typename = {"B": "binary", "I": "integer", "C": "continuous"}.get(vtype, vtype)
            print(f"  {name}: {typename}")

        ###########################################################################
        print("\n=== Constraint Matrix (A) ===")
        print("And therefore the constraint matrix shape is:", self.A.shape)
        print("Here you can see your full constraint matrix:")
        print(self.A)

        print("\n=== RHS Vector (b) ===")
        print(np.array(self.b))

        print("\n=== Senses (L ≤, G ≥, E =) ===")
        print(self.sense)

        ###########################################################################
        print("\n So your full problem consists of:")
        terms = [f"{coef}*{name}" for coef, name in zip(self.obj, self.var_names) if coef != 0]
        print("minimize")
        print("  ", " + ".join(terms))

        print("\nsubject to")
        for i in range(self.num_constraints):
            row_terms = []
            for j in range(self.num_vars):
                coeff = self.A[i, j]
                if coeff != 0:
                    row_terms.append(f"{coeff}*{self.var_names[j]}")
            expr = " + ".join(row_terms)
            rhs = self.b[i]
            sense = self.sense[i]
            if sense == 'L':
                print(f"  c{i + 1}: {expr} <= {rhs}")
            elif sense == 'G':
                print(f"  c{i + 1}: {expr} >= {rhs}")
            elif sense == 'E':
                print(f"  c{i + 1}: {expr} = {rhs}")


    def rebuild_model(self):
        import cplex

        new_model = cplex.Cplex()
        new_model.set_problem_type(cplex.Cplex.problem_type.LP)  # Or MIP if you use integer types

        # Add variables
        new_model.variables.add(
            obj=self.obj,
            lb=self.lb,
            ub=self.ub,
            types=self.var_types,
            names=self.var_names
        )

        # Add constraints
        rows = []
        senses = []
        rhs = []

        for i in range(self.A.shape[0]):
            coeffs = self.A[i, :]
            indices = np.nonzero(coeffs)[0]
            values = coeffs[indices].tolist()
            rows.append([indices.tolist(), values])
            senses.append(self.sense[i])
            rhs.append(self.b[i])

        new_model.linear_constraints.add(
            lin_expr=rows,
            senses=senses,
            rhs=rhs
        )

        self.model = new_model  # Replace the old CPLEX model

    def _clean_variable_names(self):
        new_names = [name.replace("#", "_") for name in self.var_names]
        for old, new in zip(self.var_names, new_names):
            self.model.variables.set_names([(old, new)])
        self.var_names = new_names  # Update in our local copy

    def apply_substitution_expr(self, target_var: str, const_term: float, expr_terms: List[Tuple[float, str]]):
        """
        Apply substitution of `target_var` using: target_var = const_term + sum(coeff_i * var_i)
        - const_term: constant term in the expression
        - expr_terms: list of (coefficient, variable_name)
        """
        if not hasattr(self, "obj_const"):
            self.obj_const = 0.0

        if target_var not in self.var_names:
            print(f"    [WARN] Cannot substitute: {target_var} not in model")
            return

        # Index of target_var to eliminate
        idx_target = self.var_names.index(target_var)
        coeff_target = self.obj[idx_target]

        # Update objective constant (obj += coeff_target * const_term)
        self.obj_const += coeff_target * const_term

        # Update objective coefficients: obj[var_i] += coeff_target * coeff_i
        for coeff_i, var_i in expr_terms:
            if var_i in self.var_names:
                idx_i = self.var_names.index(var_i)
                self.obj[idx_i] += coeff_target * coeff_i
            else:
                print(f"    [WARN] Substitution term {var_i} not found")

        # Update constraint matrix: A[:, var_i] += coeff_i * A[:, target_var]
        col_target = self.A[:, idx_target].copy()
        for coeff_i, var_i in expr_terms:
            if var_i in self.var_names:
                idx_i = self.var_names.index(var_i)
                self.A[:, idx_i] += coeff_i * col_target

        # Remove column for target_var
        self.A = np.delete(self.A, idx_target, axis=1)
        self.obj = np.delete(self.obj, idx_target)
        self.lb = np.delete(self.lb, idx_target)
        self.ub = np.delete(self.ub, idx_target)

        del self.var_names[idx_target]
        del self.var_types[idx_target]

        # print(f"    Substituted {target_var} → {const_term} + " +
        #       " + ".join([f"{c}*{v}" for c, v in expr_terms]))



    def build_root_model(self):
        if self.root_lp_model is not None:
            print("[WARN] root_lp_model already exists. Overwriting.")
        model = gp.Model()
        model.Params.OutputFlag = 0

        # Add variables
        x = []
        for j in range(self.num_vars):
            #vtype=self.var_types[j]
            lb = self.lb[j]
            ub = self.ub[j]
            name=self.var_names[j]
            #vartype = gp.GRB.BINARY if vtype == 'B' else gp.GRB.INTEGER if vtype == 'I' else gp.GRB.CONTINUOUS
            vartype = gp.GRB.CONTINUOUS  # default LP relaxation
            x.append(model.addVar(lb=lb, ub=ub, vtype=vartype, name=name))
        model.update()

        #Set objective
        obj_expr=gp.quicksum(self.obj[j]*x[j] for j in range(self.num_vars))
        model.setObjective(obj_expr+self.obj_const, gp.GRB.MINIMIZE)

        # Add constraints
        for i in range(self.num_constraints):
            lhs = gp.quicksum(self.A[i, j] * x[j] for j in range(self.num_vars) if self.A[i, j] != 0)
            sense = self.sense[i]
            rhs = self.b[i]
            if sense == 'L':
                model.addConstr(lhs <= rhs, name=self.row_names[i])
            elif sense == 'G':
                model.addConstr(lhs >= rhs, name=self.row_names[i])
            elif sense == 'E':
                model.addConstr(lhs == rhs, name=self.row_names[i])

        model.update()
        self.root_lp_model=model

    def is_integral(self, solution, tol=1e-6):
        """
        Check if a solution is integral on all integer or binary variables.
        """
        for i, vtype in enumerate(self.var_types):
            if vtype in ['B', 'I']:  # Only check integer/binary vars
                if abs(solution[i] - round(solution[i])) > tol:
                    return False
        return True




