from presolve.base import Presolver, Reduction
from typing import List
import numpy as np

class DualFix(Presolver):
    def __init__(self,epsilon=1e-3):
        super().__init__("DualFix")
        self.epsilon = epsilon

    def apply(self, instance) -> List[Reduction]:
        A = instance.A
        rhs=instance.b
        c=instance.obj
        sense = instance.sense
        lb = instance.lb
        ub = instance.ub
        var_names=instance.var_names
        row_names=instance.row_names
        reductions = []
        for j in range(A.shape[1]):
            col=A[:,j]
            vname=var_names[j]
            obj_coef=c[j]
            row_idxs=np.nonzero(col)[0]
            senses_involved=set(sense[i] for i in row_idxs)
            if not senses_involved.issubset({'L'}):
                continue
            col_filtered=col[row_idxs]
            if np.all(col_filtered>=0) and obj_coef>=0:
                if lb[j]>-np.inf:
                    reductions.append(Reduction('fix variable',vname,lb[j]))
                elif obj_coef-self.epsilon>0:
                    raise Exception("Your problem is unbounded or infeasible (Dual Fix check)")
                elif abs(obj_coef)<self.epsilon:
                    reductions.append(Reduction('remove_variable', vname, None))
                    for i in row_idxs:
                        cname=row_names[i]
                        reductions.append(Reduction('remove_constraint', cname, None))
            elif np.all(col_filtered<=0) and obj_coef<=0:
                if ub[j]<np.inf:
                    reductions.append(Reduction('fix variable',vname,ub[j]))
                elif obj_coef+self.epsilon<0:
                    raise Exception("Your problem is unbounded or infeasible (Dual Fix check)")
                elif abs(obj_coef)<self.epsilon:
                    reductions.append(Reduction('remove_variable', vname, None))
                    for i in row_idxs:
                        cname=row_names[i]
                        reductions.append(Reduction('remove_constraint', cname, None))
        return reductions
