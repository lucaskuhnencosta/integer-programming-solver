from bnb.node import Node
import random

class Branching:
    def __init__(self, instance,strong_depth=10,k=1500):
        self.instance = instance #Access to originalbounds, types, etc.
        self.strong_depth = strong_depth
        self.k = k

        self.pseudocosts_up=[0.0]*instance.num_vars
        self.pseudocosts_down=[0.0]*instance.num_vars
        self.pseudocounts_up=[0]*instance.num_vars
        self.pseudocounts_down=[0]*instance.num_vars
        self.score=[0]*instance.num_vars


    def _select_k_strong_candidates(self,fractional_vars):
        initialized=[]
        uninitialized=[]

        for i,val in fractional_vars:
            if self.score[i]>0:
                initialized.append((self.score[i],i))
            else:
                fractionalness = 0.5 - abs(val - 0.5)
                uninitialized.append((fractionalness,i))

        initialized.sort(reverse=True)
        uninitialized.sort(key=lambda x: x[0], reverse=True)

        selected=[i for _,i in initialized[:self.k]]

        remaining=self.k-len(selected)
        if remaining>0 and uninitialized:
            most_fractional_indices=[i for _,i in uninitialized]
            sampled=random.sample(uninitialized,min(remaining,len(uninitialized)))
            selected.extend(most_fractional_indices[:remaining])
        return selected

    def select_branching_variable(self, node, solution,working_model,active_mgr,clique_cuts=False):
        fractional_vars=[]
        for i, val in enumerate(solution):
            if self.instance.var_types[i] in ['B', 'I']:
                if abs(val - round(val)) > 1e-6:
                    fractional_vars.append((i, val))
        if clique_cuts:
            original_fractional_vars = [(i, val) for i, val in fractional_vars if i < self.instance.original_num_vars]
        else:
            original_fractional_vars = fractional_vars
        if not original_fractional_vars:
            return None
        if node.depth<=self.strong_depth:
            selected=self._select_k_strong_candidates(original_fractional_vars)
            return self.strong_branching(node,solution,selected,working_model,active_mgr)
        else:
            return self.pseudocost_branching(node,solution,original_fractional_vars,working_model,active_mgr)

    def strong_branching(self,node,solution,selected_vars,working_model,active_path):
        best_var = None
        best_score = -float('inf')
        best_down_node=None
        best_up_node=None
        original_focus = active_path.focus

        for var_idx in selected_vars:
            val=solution[var_idx]
            floor_val=int(val)
            ceil_val=floor_val+1
            lb = self.instance.lb[var_idx]
            ub = self.instance.ub[var_idx]
            down_node=Node(
                parent=node,
                bound_changes={var_idx:(lb,floor_val)},
                depth=node.depth+1
            )
            active_path.switch_focus(down_node, working_model)
            down_node.evaluate_lp(working_model,self.instance)
            obj_down = float('inf') if down_node.is_infeasible else down_node.bound
            delta_down=obj_down-node.bound

            up_node=Node(
                parent=node,
                bound_changes={var_idx:(ceil_val,ub)},
                depth=node.depth+1
            )
            active_path.switch_focus(up_node, working_model)
            up_node.evaluate_lp(working_model,self.instance)
            obj_up = float('inf') if up_node.is_infeasible else up_node.bound
            delta_up=obj_up-node.bound

            if obj_up==float('inf') or obj_down==float('inf'):
                best_var=var_idx
                best_down_node, best_up_node = down_node, up_node
                best_delta_down, best_delta_up = delta_down, delta_up
                break

            score=max(delta_down,10^-6)*max(delta_up,10^-6)
            self.score[var_idx]=score

            # 6. Keep track of the best variable with the best score
            if score>best_score:
                best_score=score
                best_var=var_idx
                best_down_node, best_up_node = down_node, up_node
                best_delta_down, best_delta_up = delta_down, delta_up

        if best_var is not None:
            val=solution[best_var]
            floor_val=int(val)
            ceil_val=floor_val+1
            f_down=val-floor_val
            f_up=ceil_val-val

            if f_down>1e-6 and best_delta_down<float('inf'):
                self.pseudocosts_down[best_var]+=best_delta_down/f_down
                self.pseudocounts_down[best_var]+=1

            if f_up>1e-6 and best_delta_up<float('inf'):
                self.pseudocosts_up[best_var]+=best_delta_up/f_up
                self.pseudocounts_up[best_var]+=1

        active_path.switch_focus(original_focus, working_model)
        return best_var, best_down_node, best_up_node

    def pseudocost_branching(self,node,solution,fractional_vars,working_model,active_path):
        best_var = None
        best_score = -float('inf')
        avg_up = self.avgg(self.pseudocosts_up, self.pseudocounts_up)
        avg_down = self.avgg(self.pseudocosts_down, self.pseudocounts_down)

        original_focus = active_path.focus  # Save current focus to restore later

        for var_idx, frac_value in fractional_vars:
            val=solution[var_idx]
            floor_val=int(val)
            ceil_val=floor_val+1

            f_down=val-floor_val
            f_up=ceil_val-val

            # Use historical data or the global average to estimate degradation
            ksi_down = (self.pseudocosts_down[var_idx] / self.pseudocounts_down[var_idx]) if self.pseudocounts_down[var_idx] > 0 else avg_down
            ksi_up = (self.pseudocosts_up[var_idx] / self.pseudocounts_up[var_idx]) if self.pseudocounts_up[var_idx] > 0 else avg_up

            score=max(f_down*ksi_down,1e-6)*max(f_up*ksi_up,1e-6)

            if score>best_score:
                best_score=score
                best_var=var_idx

        if best_var == None:
            best_var = max(fractional_vars, key=lambda item: 0.5 - abs(item[1] - 0.5))[0]

        var_idx=best_var
        val=solution[var_idx]
        floor_val, ceil_val = int(val), int(val) + 1
        lb, ub = self.instance.lb[best_var], self.instance.ub[best_var]

        best_down = Node(
            parent=node,
            bound_changes={var_idx:(lb,floor_val)},
            depth=node.depth+1)
        active_path.switch_focus(best_down, working_model)
        best_down.evaluate_lp(working_model, self.instance)

        # UP
        best_up = Node(
            parent=node,
            bound_changes={var_idx:(ceil_val,ub)},
            depth=node.depth+1)
        active_path.switch_focus(best_up, working_model)
        best_up.evaluate_lp(working_model, self.instance)

        active_path.switch_focus(original_focus, working_model)
        return best_var, best_down, best_up


    def avgg(self,costs,counts):
        values=[costs[i] / counts[i] for i in range(len(costs)) if counts[i]>0]
        return sum(values) / len(values) if values else 1
