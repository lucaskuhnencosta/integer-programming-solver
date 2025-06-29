
from presolve.base import Presolver, Reduction
from typing import List
import numpy as np

class CleanModel(Presolver):
    def __init__(self, epsilon=1e-3, psi=1e30):
        super().__init__("CleanModel")
        self.epsilon = epsilon
        self.psi = psi

    def apply(self, instance) -> List[Reduction]:
        A = instance.A
        b = instance.b
        sense = instance.sense
        lb = instance.lb
        ub = instance.ub
        var_names = instance.var_names
        row_names = instance.row_names
        reductions = []
        i=0
        while i<instance.A.shape[0]:
            row = A[i, :] #extract that row
            s = sense[i] #extract the sense
            rhs = b[i] #extract the RHS
            name = row_names[i] #extract the name
            nz = np.nonzero(row)[0]  # this returns the indices where this row in non-null
            if len(nz) == 0:
                if s=='L':
                    if rhs>=0:
                        reductions.append(Reduction('remove_constraint', name,None))
                    else:
                        raise Exception(f"Infeasible model. empty = constraint: 0 = {rhs} in {name}")
                elif s=='E':
                    if abs(rhs)<=self.epsilon:
                        reductions.append(Reduction('remove_constraint', name,None))
                    else:
                        raise Exception(f"Infeasible model. empty = constraint: 0 = {rhs} in {name}")
                i += 1
                continue
            if len(nz) == 1: #just if there is only one coefficient
                j = nz[0]  # j assumes the value of this very index
                vname=var_names[j]
                coeff = row[j]  # now we take the coefficient itself
                if s=='L':
                    if coeff > 0:
                        new_ub = rhs / coeff
                        reductions.append(Reduction('remove_constraint', name, None))
                        if new_ub < ub[j]:
                            reductions.append(Reduction('tighten_bound', vname, (lb[j], new_ub)))
                        elif new_ub < lb[j]:
                            raise Exception(f"Infeasible model. Constraint and bound of variable {vname} dont match")
                    elif coeff < 0:
                        new_lb = rhs / coeff
                        reductions.append(Reduction('remove_constraint', name, None))
                        if new_lb > lb[j]:
                            reductions.append(Reduction('tighten_bound', vname, (new_lb, ub[j])))
                        elif new_lb > ub[j]:
                            raise Exception(f"Infeasible model. Constraint and bound of variable {vname} dont match")
                elif s=='E':
                    fixed_value = rhs / coeff
                    reductions.append(Reduction('remove_constraint', name, None))

                    if fixed_value < lb[j] or fixed_value > ub[j]:
                        raise Exception(f"Infeasible model. Equality requires {vname} = {fixed_value}, "
                                        f"but bounds are [{lb[j]}, {ub[j]}]")
                    else:
                        reductions.append(Reduction('fix_variable', vname, fixed_value))

                i+=1
                continue
            # Redundant or infeasible rows
            activity_min = np.sum([row[j] * (ub[j] if row[j] < 0 else lb[j]) for j in nz])
            activity_max = np.sum([row[j] * (lb[j] if row[j] < 0 else ub[j]) for j in nz])

            if s == 'L':
                if rhs >= self.psi or activity_max <= rhs + self.epsilon:
                    reductions.append(Reduction('remove_constraint', name, None))
                elif activity_min >= rhs + self.epsilon:
                    raise Exception(f"Infeasible row detected: {name}")
            elif s == 'E':
                if (activity_min > rhs + self.epsilon) or (activity_max < rhs - self.epsilon):
                    raise Exception(f"Infeasible equality: {name}")
                elif abs(activity_min - rhs) <= self.epsilon and abs(activity_max - rhs) <= self.epsilon:
                    reductions.append(Reduction('remove_constraint', name, None))
            i+=1

        # Fix variables where lb ≈ ub

        for name, l, u in zip(var_names,lb,ub):
            if abs(l - u) < self.epsilon:
                reductions.append(Reduction('fix_variable', name, l))

        for j in range(len(var_names)):
            column=A[:,j]
            column_nmz = np.count_nonzero(column, axis=0)
            vname = var_names[j]
            coef = instance.obj[j]
            if column_nmz == 0:
                if abs(coef)<self.epsilon:
                    #print(f"f    removing unused variable {vname}")
                    reductions.append(Reduction('remove_variable', vname, None))
                elif coef>0:
                    fix_val=lb[j]
                    #print(f"    Fixing objective-only variable {vname} = {fix_val} (obj>0)")
                    reductions.append(Reduction(kind='fix_variable', target=vname, value=fix_val))
                else:
                    fix_val=ub[j]
                    #print(f"    Fixing objective-only variable {vname} = {fix_val} (obj<0)")
                    reductions.append(Reduction(kind='fix_variable', target=vname, value=fix_val))


        # Ensure integral bounds for integer variables
        for j, (vtype, name) in enumerate(zip(instance.var_types, var_names)):
            if vtype in ('I', 'B'):  # Integer or Binary
                orig_lb = lb[j]
                orig_ub = ub[j]

                int_lb = np.ceil(orig_lb - self.epsilon)
                int_ub = np.floor(orig_ub + self.epsilon)

                # Infeasibility check after rounding
                if int_lb > int_ub + self.epsilon:
                    raise Exception(f"Infeasible integer bounds for variable {name} after rounding.")

                changed = False
                if abs(orig_lb - int_lb) > self.epsilon:
                    lb[j] = int_lb
                    changed = True
                if abs(orig_ub - int_ub) > self.epsilon:
                    ub[j] = int_ub
                    changed = True

                if changed:
                    #print(f"    Adjusted bounds for integer variable {name}: [{orig_lb}, {orig_ub}] → [{int_lb}, {int_ub}]")
                    reductions.append(Reduction('tighten_bound', name, (lb[j], ub[j])))


        return reductions