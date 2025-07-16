from presolve.base import Presolver, Reduction
from typing import List
import numpy as np

class ColSingletonRemover(Presolver):
    def __init__(self):
        super().__init__("ColSingletonRemover")

    def apply(self, instance) -> List[Reduction]:
        A = instance.A
        b=instance.b
        sense=instance.sense
        var_names=instance.var_names
        row_names=instance.row_names
        reductions = []
        for j in range(A.shape[1]):
            col = A[:, j]
            row_indices=np.flatnonzero(col)
            if len(row_indices)==1: #If it is a singleton column
                i=row_indices[0] #The only row this singleton variable appears in
                if sense[i] != 'E':
                    continue
                row = A[i, :]
                if len(np.flatnonzero(row))!=2:
                    continue
                coeff_j=A[i,j]
                rhs=b[i]
                x_to_sub = var_names[j]
                terms=[]
                for k in np.flatnonzero(row):
                    if k!=j:
                        coeff_k=A[i,k]
                        other_var=var_names[k]
                        terms.append((-coeff_k/coeff_j,other_var))
                const_term=rhs/coeff_j
                reductions.append(Reduction(kind='substitute_variable',target=x_to_sub,value=(const_term,terms)))
                reductions.append(Reduction(kind='remove_constraint',target=row_names[i],value=None))
                print("Remove constraint comes from Singleton Column ")
        return reductions