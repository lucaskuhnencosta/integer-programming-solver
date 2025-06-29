from presolve.base import Presolver, Reduction
from typing import List
import numpy as np

class BoundTightener(Presolver):
    def __init__(self, epsilon=1e-3):
        super().__init__("BoundTightener")
        self.epsilon = epsilon

    def apply(self, instance) -> List[Reduction]:
        A = instance.A
        rhs = instance.b
        sense = instance.sense
        lb = instance.lb
        ub = instance.ub
        var_names = instance.var_names
        var_types = instance.var_types
        row_names = instance.row_names

        reductions = []

        for i in range(A.shape[0]):
            row_sense = sense[i]
            if row_sense == 'L':
                row = A[i, :]
                row_rhs = rhs[i]
                min_activity = 0.0
                max_activity = 0.0
                for j in range(A.shape[1]):
                    coeff = row[j]
                    if coeff > 0:
                        min_activity += coeff * lb[j]
                        max_activity += coeff * ub[j]
                    elif coeff < 0:
                        min_activity += coeff * ub[j]
                        max_activity += coeff * lb[j]

                for j in range(A.shape[1]):
                    coeff = row[j]
                    if coeff == 0:
                        continue
                    x_name = var_names[j]
                    if coeff > 0:
                        new_ub = (row_rhs - (min_activity - coeff * lb[j])) / coeff
                        if new_ub < ub[j] - self.epsilon:
                            #print(f"    Tightened bounds for {x_name}: [{new_ub}, {ub[j]}]")
                            reductions.append(Reduction('tighten_bound', x_name, (lb[j], new_ub)))
                    elif coeff < 0:
                        new_lb = (row_rhs - (min_activity - coeff * ub[j])) / coeff
                        if new_lb > lb[j] + self.epsilon:
                            #print(f"    Tightened bounds for {x_name}: [{lb[j]}, {new_lb}]")
                            reductions.append(Reduction('tighten_bound', x_name, (new_lb, ub[j])))

        return reductions
