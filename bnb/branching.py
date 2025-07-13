from bnb.node import Node
import random

class Branching:
    def __init__(self, instance,strong_depth=100,k=1500):
        self.instance = instance #Access to originalbounds, types, etc.
        self.strong_depth = strong_depth
        self.k = k

        self.pseudocosts_up=[0.0]*instance.num_vars
        self.pseudocosts_down=[0.0]*instance.num_vars
        self.pseudocounts_up=[0]*instance.num_vars
        self.pseudocounts_down=[0]*instance.num_vars
        self.score=[0]*instance.num_vars


    def _select_k_strong_candidates(self,fractional_vars):
        """
        From all fractional vars, select up to k:
        - Prefer vars with initialized pseudocosts in both directions.
        - Fill remaining slots randomly from the rest.
        """
        initialized=[]
        uninitialized=[]

        for i in fractional_vars:
            if self.score[i]>0:
                initialized.append((self.score[i],i))
            else:
                uninitialized.append(i)

        initialized.sort(reverse=True)

        selected=[i for _,i in initialized[:self.k]]
        remaining=self.k-len(selected)

        if remaining>0 and uninitialized:
            sampled=random.sample(uninitialized,min(remaining,len(uninitialized)))
            selected.extend(sampled)

        return selected

    def select_branching_variable(self, node, solution,working_model,active_mgr,strong_k_override=None):
        """
        Given a fractional solution from the current LP relaxation (at a node),
        evaluate all fractional integer variables with strong branching.
        This means solving both child LPs that would result from fixing the variable down and up,
        and picking the variable whose branching seems most promising.
        """
        # 1. Identify all fractional variables that must be branched on
        fractional_vars=[
            i for i,val in enumerate(solution)
            if self.instance.var_types[i] in ['B','I'] and abs(val-round(val))>1e-6
        ]

        if not fractional_vars:
            return None

        if node.depth<=self.strong_depth:
            self.k=strong_k_override if strong_k_override is not None else self.k
            selected=self._select_k_strong_candidates(fractional_vars)
            return self.strong_branching(node,solution,selected,working_model,active_mgr)
        else:
            return self.pseudocost_branching(node,solution,fractional_vars,working_model,active_mgr)

    def strong_branching(self,node,solution,selected_vars,working_model,active_path):
        best_var = None  # The best variable to branch on
        best_score = -float('inf')  # Highest strong branching score so far
        best_down=None
        best_up=None
        original_focus = active_path.focus  # Save current focus to restore later

        for var_idx in selected_vars:
            val=solution[var_idx]
            floor_val=int(val) # e.g., 2.3 → floor_val = 2
            ceil_val=floor_val+1 # e.g., 2.3 → ceil_val = 3
            #Update pseudocosts
            f_down=val-floor_val
            f_up=ceil_val-val

            # 2. Look up the *original global bounds* for this variable (see next explanation)
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

            up_node=Node(
                parent=node,
                bound_changes={var_idx:(ceil_val,ub)},
                depth=node.depth+1
            )
            active_path.switch_focus(up_node, working_model)
            up_node.evaluate_lp(working_model,self.instance)
            obj_up = float('inf') if up_node.is_infeasible else up_node.bound

            # 5. Use the best bound you'd get from either branch as a score
            # (You could use other heuristics here, like |obj_up - obj_down|.)
            score=min(obj_down,obj_up)
            self.score[var_idx]=score

            # 6. Keep track of the best variable with the best score
            if score>best_score:
                best_score=score
                best_var=var_idx
                best_down=down_node
                best_up=up_node

            if obj_down <float('inf') and f_down>1e-6:
                delta_down=obj_down-node.bound
                self.pseudocosts_down[var_idx]+=delta_down/f_down
                self.pseudocounts_down[var_idx]+=1

            if obj_up<float('inf') and f_up>1e-6:
                delta_up=obj_up-node.bound
                self.pseudocosts_up[var_idx]+=delta_up/f_up
                self.pseudocounts_up[var_idx]+=1

        active_path.switch_focus(original_focus, working_model)
                
        return best_var, best_down, best_up

    def pseudocost_branching(self,node,solution,fractional_vars,working_model,active_path):
        best_var = None  # The best variable to branch on
        best_score = -float('inf')  # Highest strong branching score so far

        avg_up = self.avgg(self.pseudocosts_up, self.pseudocounts_up)
        avg_down = self.avgg(self.pseudocosts_down, self.pseudocounts_down)

        original_focus = active_path.focus  # Save current focus to restore later

        for var_idx in fractional_vars:
            val=solution[var_idx]
            floor_val=int(val)
            ceil_val=floor_val+1

            f_down=val-floor_val
            f_up=ceil_val-val

            eta_down=self.pseudocounts_down[var_idx]
            eta_up=self.pseudocounts_up[var_idx]

            if eta_down>0 and eta_up>0:
                ksi_down=self.pseudocosts_down[var_idx]/eta_down
                ksi_up=self.pseudocosts_up[var_idx]/eta_up
            else:
                ksi_down=avg_down
                ksi_up=avg_up

            score=(5/6)*min(f_down*ksi_down,f_up*ksi_up)+(1/6)*max(f_down*ksi_down,f_up*ksi_up)

            if score>best_score:
                best_score=score
                best_var=var_idx

                lb=self.instance.lb[var_idx]
                ub=self.instance.ub[var_idx]

                # DOWN
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
