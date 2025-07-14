import cplex
import numpy as np
import gurobipy as gp
from typing import List, Tuple, Dict
import os # Make sure os is imported at the top of the file


class MIPInstance:
    def __init__(self, mps_path: str):
        self.mps_path = mps_path

        self.obj_const=0
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
        filename = os.path.basename(mps_path)
        if filename.startswith("instance"):
            self.sense_obj = 1
        else:
            self.sense_obj = -1
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
        # self.sense_obj = self.model.objective.get_sense()  # <-- ADD THIS
        # print("\n" + "=" * 50)
        # print(f"DEBUG: CPLEX read objective sense value as: {self.sense_obj}")
        # print("(Note: 1 = Minimize, -1 = Maximize)")
        # print("=" * 50 + "\n")

        if self.sense_obj == -1:  # -1 signifies maximization
            print("[INFO] Maximization problem detected. Negating objective function.")
            self.obj = [-c for c in self.obj]
            # Also negate the constant, if it ever exists from presolve
            self.obj_const = -self.obj_const
            # Now, we can treat it as a minimization problem everywhere else
            self.sense_obj = 1

            ### Variables, their types, bounds, names, and number ###
        self.var_types = self.model.variables.get_types()
        self.lb = self.model.variables.get_lower_bounds()
        self.ub = self.model.variables.get_upper_bounds()
        self.var_names = self.model.variables.get_names()
        self._clean_variable_names()


        # Map CPLEX variable indices to our internal dense index
        self.var_index_map = {name: idx for idx, name in enumerate(self.var_names)}
        self.reverse_var_index = {v: k for k, v in self.var_index_map.items()}

        num_sos = self.model.SOS.get_num()
        if num_sos > 0:
            print("\n" + "=" * 60)
            print(f"⚠️  WARNING: This model contains {num_sos} Special Ordered Set (SOS) constraint(s).")
            print("   This custom solver does not handle SOS constraints explicitly.")
            print("   The solution may be incorrect as these constraints will be ignored.")
            print("=" * 60 + "\n")
        # --- End of new code ---

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
        # print("\n=== Objective ===")
        # print("Objective vector:", self.obj)

        ###########################################################################
        print('\n=== Variables ===')

        print(f"Total binary variables: {self.num_binary}")
        print(f"Total integer variables: {self.num_integer}")
        print(f"Total continuous variables: {self.num_continuous}")

        # print("Variable types:", self.var_types)
        # print("Variable names:", self.var_names)
        # print("Bounds:")
        # for name, lb, ub in zip(self.var_names, self.lb, self.ub):
        #     if abs(lb - ub) < 1e-8:
        #         print(f"  {name}: FIXED to {lb}")
        #     else:
        #         lb_str = str(lb) if lb > -1e20 else "-∞"
        #         ub_str = str(ub) if ub < 1e20 else "∞"
        #         print(f"  {name}: [{lb_str}, {ub_str}]")
        #
        # print("Variable Types:")
        # for name, vtype in zip(self.var_names, self.var_types):
        #     typename = {"B": "binary", "I": "integer", "C": "continuous"}.get(vtype, vtype)
        #     print(f"  {name}: {typename}")

        # ###########################################################################
        # print("\n=== Constraint Matrix (A) ===")
        # print("And therefore the constraint matrix shape is:", self.A.shape)
        # print("Here you can see your full constraint matrix:")
        # print(self.A)
        #
        # print("\n=== RHS Vector (b) ===")
        # print(np.array(self.b))
        #
        # print("\n=== Senses (L ≤, G ≥, E =) ===")
        # print(self.sense)
        #
        # ###########################################################################
        # print("\n So your full problem consists of:")
        # terms = [f"{coef}*{name}" for coef, name in zip(self.obj, self.var_names) if coef != 0]
        # print("minimize")
        # print("  ", " + ".join(terms))
        #
        # print("\nsubject to")
        # for i in range(self.num_constraints):
        #     row_terms = []
        #     for j in range(self.num_vars):
        #         coeff = self.A[i, j]
        #         if coeff != 0:
        #             row_terms.append(f"{coeff}*{self.var_names[j]}")
        #     expr = " + ".join(row_terms)
        #     rhs = self.b[i]
        #     sense = self.sense[i]
        #     if sense == 'L':
        #         print(f"  c{i + 1}: {expr} <= {rhs}")
        #     elif sense == 'G':
        #         print(f"  c{i + 1}: {expr} >= {rhs}")
        #     elif sense == 'E':
        #         print(f"  c{i + 1}: {expr} = {rhs}")


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

    def _complement_all_binary_vars(self):
        """
        For every binary variable x, creates a complement x_comp and adds the
        linking constraint x + x_comp = 1. It then uses these complements
        to eliminate all negative coefficients for binary variables in the model.
        """
        # print("⚙️  Complementing all binary variables to handle negative coefficients...")

        # 1. Identify all original binary variables
        original_binary_indices = [i for i, vtype in enumerate(self.var_types) if vtype == 'B']
        if not original_binary_indices:
            # print("    No binary variables to complement.")
            return

        # Store original counts before we start modifying the model
        self.original_num_vars = self.num_vars
        self.original_num_constrs = self.num_constraints

        # Create a mapping from original variable index to its new complement's index
        complement_map = {}

        # 2. Add a new complement variable for each original binary variable
        for j in original_binary_indices:
            original_name = self.var_names[j]
            complement_name = f"{original_name}_comp"

            # Add the new variable's properties
            self.var_names.append(complement_name)
            self.var_types.append('B')
            self.lb = np.append(self.lb, 0.0)
            self.ub = np.append(self.ub, 1.0)
            self.obj = np.append(self.obj, 0.0)

            # Add a new column of zeros to the A matrix for the new variable
            new_col = np.zeros((self.A.shape[0], 1))
            self.A = np.hstack([self.A, new_col])

            # Store the index of the newly created complement variable
            complement_map[j] = self.num_vars - 1

        # 3. Add linking constraints (x + x_comp = 1) for all pairs
        for j in original_binary_indices:
            link_row = np.zeros((1, self.num_vars))
            link_row[0, j] = 1.0
            link_row[0, complement_map[j]] = 1.0

            self.A = np.vstack([self.A, link_row])
            self.b = np.append(self.b, 1.0)
            self.sense = np.append(self.sense, 'E')
            self.row_names = np.append(self.row_names, f"link_{self.var_names[j]}")

        # 4. Perform the substitution: replace `-a*x` with `a*(1 - x_comp)`
        # This is equivalent to: `a - a*x_comp`
        for i in range(self.original_num_constrs):
            for j in original_binary_indices:
                coeff = self.A[i, j]
                if coeff < 0:
                    # The term is -a_ij * x_j. Substitute x_j = 1 - x_j_comp
                    # The term becomes -a_ij * (1 - x_j_comp) = -a_ij + a_ij * x_j_comp

                    # Add the constant term `-a_ij` to the RHS
                    self.b[i] -= coeff

                    # Add the new variable term `+a_ij * x_j_comp`
                    complement_idx = complement_map[j]
                    self.A[i, complement_idx] += -coeff  # Add the positive coefficient

                    # Remove the original negative term by setting its coefficient to 0
                    self.A[i, j] = 0.0

        # print(f"    Added {len(original_binary_indices)} complement variables and linking constraints.")

    def get_binary_subproblem(self):
        """
        Filters the main problem to find constraints that only involve binary variables,
        which are candidates for clique detection
        """
        binary_indices = {i for i, vtype in enumerate(self.var_types) if vtype == 'B'}

        positive_binary_constraints = []
        positive_binary_rhs = []
        total_binary_constraints = 0

        for i in range(self.num_constraints):
            # Find the indices of all variables with non-zero coefficients in this row
            vars_in_row = set(np.nonzero(self.A[i, :])[0])

            # If all variables in the constraint are binary, it's a candidate
            if vars_in_row and vars_in_row.issubset(binary_indices):
                total_binary_constraints +=1
                constraint_terms = [(j, self.A[i, j]) for j in vars_in_row]

                if all(coeff > 0 for _, coeff in constraint_terms):
                    positive_binary_constraints.append(constraint_terms)
                    positive_binary_rhs.append(self.b[i])


        print(f"    Found {total_binary_constraints} constraints involving only binary variables.")
        print(f"    Filtered down to {len(positive_binary_constraints)} constraints with all-positive coefficients.")

        # Verification step to confirm our logic is correct
        all_coeffs_are_positive = all(
            coeff > 0
            for constraint in positive_binary_constraints
            for _, coeff in constraint
        )
        print(f"    Verification check: All coefficients are positive? -> {all_coeffs_are_positive}")
        if not all_coeffs_are_positive:
            print("    [WARNING] Verification failed. A non-positive coefficient was found.")

        return positive_binary_constraints, positive_binary_rhs






