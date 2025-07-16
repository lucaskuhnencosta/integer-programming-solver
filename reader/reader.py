import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import List, Tuple, Dict
import os

class MIPInstance:
    def __init__(self, mps_path: str):
        self.mps_path = mps_path
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        self.model = gp.read(mps_path, env=env)
        filename = os.path.basename(mps_path)
        if filename.startswith("instance"):
            self.sense_obj = 1
        else:
            self.sense_obj = -1
        self.A = None
        self.b = []
        self.sense = []
        self.lb = []
        self.ub = []
        self.var_types = []
        self.var_names = []
        self.obj = []
        self.obj_const = 0
        self.root_lp_model=None

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
        self.model.update()

        ### Objective ###
        # if self.model.ModelSense == GRB.MAXIMIZE or self.sense_obj == -1:
        #     print("[INFO] Maximization problem detected. Negating objective function.")
        #     self.model.ModelSense = GRB.MINIMIZE
        #     for v in self.model.getVars():
        #         v.Obj = -v.Obj
        #     self.obj_const = -self.model.ObjCon
        #     self.model.setObjective(self.model.getObjective() + self.obj_const)

        # Extract data from the (now guaranteed to be MINIMIZE) Gurobi model
        variables = self.model.getVars()
        self.var_names = [v.VarName for v in variables]
        self.obj = np.array([v.Obj for v in variables])
        self.lb = np.array([v.LB for v in variables])
        self.ub = np.array([v.UB for v in variables])
        self.var_types = [v.Vtype for v in variables]
        constraints = self.model.getConstrs()
        self.row_names = np.array([c.ConstrName for c in constraints])
        self.b = np.array([c.RHS for c in constraints])
        sense_map = {GRB.LESS_EQUAL: 'L', GRB.GREATER_EQUAL: 'G', GRB.EQUAL: 'E'}
        self.sense = [sense_map[c.Sense] for c in constraints]
        self.A = self.model.getA().toarray()

        if self.sense_obj == -1:
            self.obj = -self.obj
            self.obj_const = -self.obj_const

    def pretty_print(self):
        print(f'This model has originally {self.num_vars} variables and {self.num_constraints} constraints')
        print('\n=== Variables ===')
        print(f"Total binary variables: {self.num_binary}")
        print(f"Total integer variables: {self.num_integer}")
        print(f"Total continuous variables: {self.num_continuous}")

    def _clean_variable_names(self):
        new_names = [name.replace("#", "_") for name in self.var_names]
        for old, new in zip(self.var_names, new_names):
            self.model.variables.set_names([(old, new)])
        self.var_names = new_names  # Update in our local copy

    def apply_substitution_expr(self, target_var: str, const_term: float, expr_terms: List[Tuple[float, str]]):
        if not hasattr(self, "obj_const"):
            self.obj_const = 0.0
        if target_var not in self.var_names:
            print(f"    [WARN] Cannot substitute: {target_var} not in model")
            return
        idx_target = self.var_names.index(target_var)
        coeff_target = self.obj[idx_target]
        self.obj_const += coeff_target * const_term
        for coeff_i, var_i in expr_terms:
            if var_i in self.var_names:
                idx_i = self.var_names.index(var_i)
                self.obj[idx_i] += coeff_target * coeff_i
            else:
                print(f"    [WARN] Substitution term {var_i} not found")
        col_target = self.A[:, idx_target].copy()
        self.b-=col_target*const_term
        for coeff_i, var_i in expr_terms:
            if var_i in self.var_names:
                idx_i = self.var_names.index(var_i)
                self.A[:, idx_i] += coeff_i * col_target
        self.A = np.delete(self.A, idx_target, axis=1)
        self.obj = np.delete(self.obj, idx_target)
        self.lb = np.delete(self.lb, idx_target)
        self.ub = np.delete(self.ub, idx_target)
        del self.var_names[idx_target]
        del self.var_types[idx_target]

    def build_root_model(self):
        if self.root_lp_model is not None:
            print("[WARN] root_lp_model already exists. Overwriting.")
        model = gp.Model()
        model.Params.OutputFlag = 0
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
        obj_expr=gp.quicksum(self.obj[j]*x[j] for j in range(self.num_vars))
        model.setObjective(obj_expr+self.obj_const, gp.GRB.MINIMIZE)
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

        # In reader/reader.py, add this to the MIPInstance class

        # In reader.py, inside the MIPInstance class

    def build_gurobi_model(self):
        """Builds and returns a Gurobi model from the current instance data."""
        model = gp.Model()
        # model.Params.OutputFlag = 0

        # Add variables
        vars = model.addVars(self.num_vars,
                             lb=self.lb,
                             ub=self.ub,
                             vtype=self.var_types,
                             name=self.var_names)

        model.update()

        # Add constraints
        for i in range(self.num_constraints):
            lhs = gp.quicksum(self.A[i, j] * vars[j] for j in range(self.num_vars) if self.A[i, j] != 0)
            sense_char = self.sense[i]
            sense = GRB.LESS_EQUAL if sense_char == 'L' else (GRB.GREATER_EQUAL if sense_char == 'G' else GRB.EQUAL)
            model.addConstr(lhs, sense, self.b[i], name=self.row_names[i])

        # Set objective
        objective_expression = gp.quicksum(self.obj[j] * vars[j] for j in range(self.num_vars))
        model.setObjective(objective_expression + self.obj_const, GRB.MINIMIZE)

        return model

    def write_model(self, output_path):
        """Builds a Gurobi model and writes it to a file."""
        print(f"  LOG: Writing current model state to {output_path}...")
        model = self.build_gurobi_model()
        model.write(output_path)






