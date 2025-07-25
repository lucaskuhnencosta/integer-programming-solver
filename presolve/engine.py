from typing import List
from collections import Counter
from presolve.base import Presolver, Reduction
import numpy as np

class PresolveEngine:
    def __init__(self, instance):
        self.instance = instance
        self.presolvers: List[Presolver] = []
        self.applied_reductions: List[Reduction] = []

    def register(self, presolver: Presolver):
        self.presolvers.append(presolver)

    def run(self, max_rounds: int = 10):
        round_count = 0
        changed = True
        while changed and round_count < max_rounds:
            changed = False
            round_count += 1
            # print(f"Presolve Round {round_count}")
            for presolver in self.presolvers:
                reductions = presolver.apply(self.instance)
                if reductions:
                    self._apply_reductions(reductions)
                    changed = True

    def _apply_reductions(self, reductions: List[Reduction]):
        for r in reductions:
            if r.kind == 'tighten_bound':
                vname=r.target
                try:
                    idx=self.instance.var_names.index(vname)
                    old_lb,old_ub=self.instance.lb[idx],self.instance.ub[idx]
                    new_lb,new_ub=r.value
                    if new_lb > new_ub + 1e-6:
                        raise Exception(f"Infeasible tightening for {vname}: [{new_lb}, {new_ub}]")
                    if new_lb > old_lb + 1e-6 or new_ub < old_ub - 1e-6:
                        self.instance.lb[idx] = max(old_lb, new_lb)
                        self.instance.ub[idx] = min(old_ub, new_ub)
                        self.applied_reductions.append(r)
                except ValueError:
                    print(f"    Failed to tighten bounds for {vname}: not found")
            elif r.kind == 'tighten_coefficient':
                row_name, var_name = r.target
                old, new = r.value
                try:
                    row_idx = np.where(self.instance.row_names == row_name)[0][0]
                    col_idx = self.instance.var_names.index(var_name)
                    self.instance.A[row_idx, col_idx] = new
                    self.applied_reductions.append(r)
                except ValueError:
                    print(f"    Skipped tightening: row {row_name} or variable {var_name} no longer exists")
            elif r.kind == 'remove_constraint':
                cname = r.target
                try:
                    idx = np.where(self.instance.row_names == cname)[0][0]
                    self.instance.A=np.delete(self.instance.A,idx,0)
                    self.instance.b=np.delete(self.instance.b,idx)
                    self.instance.sense=np.delete(self.instance.sense,idx)
                    self.instance.row_names=np.delete(self.instance.row_names,idx)
                    self.applied_reductions.append(r)
                except Exception as e:
                    print(f"Failed to remove empty constraint {cname}: {e}")
            elif r.kind=='remove_variable':
                vname = r.target
                try:
                    idx=self.instance.var_names.index(vname)
                    self.instance.A=np.delete(self.instance.A,idx,1)
                    self.instance.obj=np.delete(self.instance.obj,idx)
                    self.instance.lb=np.delete(self.instance.lb,idx)
                    self.instance.ub=np.delete(self.instance.ub,idx)
                    del self.instance.var_names[idx]
                    del self.instance.var_types[idx]
                    self.applied_reductions.append(r)
                except ValueError:
                    print(f"    Failed to remove variable {vname}: not found")
            elif r.kind == 'fix_variable':
                vname = r.target
                try:
                    idx = self.instance.var_names.index(vname)
                    value=r.value
                    self.instance.b-=self.instance.A[:,idx]*value
                    if hasattr(self.instance, "obj_const"):
                        self.instance.obj_const += self.instance.obj[idx] * value
                    else:
                        self.instance.obj_const = self.instance.obj[idx] * value
                    self.instance.obj[idx] = 0.0  # Optional: mark as 0
                    self.instance.A=np.delete(self.instance.A,idx,axis=1)
                    self.instance.obj=np.delete(self.instance.obj,idx)
                    self.instance.lb=np.delete(self.instance.lb,idx)
                    self.instance.ub=np.delete(self.instance.ub,idx)
                    del self.instance.var_names[idx]
                    del self.instance.var_types[idx]
                    self.applied_reductions.append(r)
                except ValueError:
                    print(f"    Failed to fix variable {vname}: not found")
            elif r.kind=='substitute_variable':
                const_term,expr_terms=r.value
                target=r.target
                self.instance.apply_substitution_expr(target,const_term,expr_terms)
                self.applied_reductions.append(r)

    def summary(self):
        print("Presolve complete. Reductions applied:")
        counter=Counter(r.kind for r in self.applied_reductions)
        for kind,count in counter.items():
            print(f"{kind}: {count}")
