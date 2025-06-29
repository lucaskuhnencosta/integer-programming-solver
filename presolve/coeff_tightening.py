from presolve.base import Presolver, Reduction
from typing import List
import numpy as np

class CoefficientTightening(Presolver):
    def __init__(self):
        super().__init__("CoefficientTightening")

    def apply(self, instance) -> List[Reduction]:
        A = instance.A
        sense = instance.sense
        rhs = instance.b
        lb = instance.lb
        ub = instance.ub
        reductions = []
        row_names = instance.row_names
        var_names = instance.var_names

        for i in range(A.shape[0]):
            row = A[i, :]
            row_sense = sense[i]
            row_rhs = rhs[i]
            row_name = row_names[i]

            activity_min = 0.0
            activity_max = 0.0
            for j in range(len(var_names)):
                a_ij = row[j]
                if a_ij > 0:
                    activity_min += a_ij * lb[j]
                    activity_max += a_ij * ub[j]
                elif a_ij < 0:
                    activity_min += a_ij * ub[j]
                    activity_max += a_ij * lb[j]

            for j in np.nonzero(row)[0]:
                a_ij = row[j]
                xj = var_names[j]
                old_aij = a_ij

                # Try to shrink a_ij to a_ij' such that feasibility is preserved
                if row_sense == 'L':
                    if a_ij > 0 and ub[j] < 1e20:
                        slack = row_rhs - activity_max - a_ij * (ub[j]-1)
                        new_aij = min(a_ij, a_ij-slack)
                    elif a_ij < 0 and lb[j] > -1e20:
                        slack = row_rhs - activity_max - a_ij * (lb[j]+1)
                        new_aij= max(a_ij, a_ij+slack)
                    else:
                        continue
                else:
                    continue

                if abs(new_aij - a_ij) > 1e-8:
                    A[i, j] = new_aij
                    reductions.append(Reduction('tighten_coefficient', (row_name, xj), (old_aij, new_aij)))

        return reductions

