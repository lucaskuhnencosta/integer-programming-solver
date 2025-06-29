import csv
import time
import os
import gurobipy as gp
import numpy as np
from collections import Counter
#np.set_printoptions(threshold=np.inf,linewidth=np.inf)

from reader.reader import MIPInstance

from presolve.ModelCanonicalizer import ModelCanonicalizer

from presolve.engine import PresolveEngine

from presolve.clean_model import CleanModel
from presolve.singleton_cols import ColSingletonRemover
from presolve.bound_tightening import BoundTightener
from presolve.coeff_tightening import CoefficientTightening
from presolve.dual_fix import DualFix


#INPUT_DIR_A = "MIPlib_instances"
INPUT_DIR_B = "small_instances"
OUTPUT_DIR = "small_instances_simplified_model"
LOG_PATH = "results.csv"



def presolve_and_save(instance: MIPInstance, out_path):
    ModelCanonicalizer().apply(instance)
    engine = PresolveEngine(instance)

    engine.register(CleanModel())
    engine.register(ColSingletonRemover())
    engine.register(BoundTightener())
    engine.register(CoefficientTightening())
    engine.register(DualFix())
    engine.run()
    engine.summary()

    instance.rebuild_model()

    #Save as .mps
    instance.model.write(out_path)

    # Save as .lp
    lp_path = out_path.replace(".mps", ".lp")
    instance.model.write(lp_path)

    simplified_instance = MIPInstance(out_path)

    return simplified_instance, engine.applied_reductions

def main():
    with open(LOG_PATH, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["instance",
                     "cons_before", "cons_after",
                     "bin_vars_before","bin_vars_after",
                     "Int_variables_before", "Int_variables_after",
                     "Cont_variables_before", "Cont_variables_after",
                        "obj_original","obj_reduced",
                     "Solve time(s) before","Solve time(s) after",
                     "Bound","Coeff","remove_cons","remove var","fix_var","subst_var"])

        for fname in os.listdir(INPUT_DIR_B):
            if not fname.endswith(".mps"):
                continue
            full_path = os.path.join(INPUT_DIR_B, fname)
            reduced_path = os.path.join(OUTPUT_DIR, fname)

            print(f"============ Processing {fname} ============")
            instance_before = MIPInstance(full_path)
            #instance_before.pretty_print() #enable if want to look at instance
            vars_before = instance_before.num_vars
            cons_before = instance_before.num_constraints
            bin_vars_before=instance_before.num_binary
            int_vars_before=instance_before.num_integer
            continuous_vars_before=instance_before.num_continuous

            obj_original,solve_time_original = solve_with_gurobi(full_path)

            simplified_instance, applied_reductions = presolve_and_save(instance_before, reduced_path)
            counts = count_reductions_by_type(applied_reductions)
            tighten_bound = counts['tighten_bound']
            tighten_coefficient = counts['tighten_coefficient']
            remove_constraint = counts['remove_constraint']
            remove_variable = counts['remove_variable']
            fix_variable = counts['fix_variable']
            substitute_variable = counts['substitute_variable']

            obj_reduced,solve_time_reduced = solve_with_gurobi(reduced_path)

            vars_after = simplified_instance.num_vars
            bin_vars_after=simplified_instance.var_types.count('B')
            int_vars_after=simplified_instance.var_types.count('I')
            continuous_vars_after=simplified_instance.var_types.count('C')
            cons_after=simplified_instance.num_constraints

            print_summary_table(vars_before, vars_after,
                                cons_before, cons_after,
                                bin_vars_before, bin_vars_after,
                                int_vars_before, int_vars_after,
                                continuous_vars_before, continuous_vars_after,
                                obj_original, obj_reduced,
                                solve_time_original,solve_time_reduced)

            summarize_reductions(applied_reductions)

            writer.writerow([fname,
                             vars_before, vars_after,
                             cons_before, cons_after,
                             bin_vars_before, bin_vars_after,
                             int_vars_before, int_vars_after,
                             continuous_vars_before, continuous_vars_after,
                             obj_original, obj_reduced,
                             solve_time_original, solve_time_reduced,
                             tighten_bound, tighten_coefficient,
                             remove_constraint, remove_variable,
                             fix_variable, substitute_variable])




def print_summary_table(vars_before, vars_after,
                        cons_before, cons_after,
                        bin_vars_before, bin_vars_after,
                        int_vars_before, int_vars_after,
                        continuous_vars_before, continuous_vars_after,
                        obj_original, obj_reduced,
                        solve_time_original, solve_time_reduced):

    print("\n\n========== Pre-Solve Summary ==========\n")
    print(f"{'':35} | {'Original':>10} | {'Simplified':>10}")
    print("-" * 62)
    print(f"{'Constraints':35} | {cons_before:10} | {cons_after:10}")
    print(f"{'variables':35} | {vars_before:10} | {vars_after:10}")
    print(f"{'Binary Variables':35} | {bin_vars_before:10} | {bin_vars_after:10}")
    print(f"{'Integer Variables':35} | {int_vars_before:10} | {int_vars_after:10}")
    print(f"{'Continuous Variables':35} | {continuous_vars_before:10} | {continuous_vars_after:10}")
    print(f"{'Objective Value':35} | {obj_original:10.4f} | {obj_reduced:10.4f}")
    print(f"{'Solve Time (s)':35} | {solve_time_original:10.4f} | {solve_time_reduced:10.4f}")
    print("-" * 62)





def solve_with_gurobi(mps_path):
    model = gp.read(mps_path)
    # model.setParam("Presolve", 0)
    # model.setParam("OutputFlag", 0)
    start = time.time()
    model.optimize()
    end = time.time()
    obj=model.ObjVal if model.SolCount>0 else None
    return obj, end - start

def summarize_reductions(applied_reductions):
    counter = Counter(r.kind for r in applied_reductions)
    print("\nReduction Summary:")
    print(f"{'Reduction Type':<25} | Count")
    print("-" * 40)
    for kind, count in counter.items():
        print(f"{kind:<25} | {count}")


def count_reductions_by_type(applied_reductions):
    # Known reduction types
    reduction_types = [
        'tighten_bound',
        'tighten_coefficient',
        'remove_constraint',
        'remove_variable',
        'fix_variable',
        'substitute_variable'
    ]
    counter = Counter(r.kind for r in applied_reductions)

    # Initialize results for all known types
    counts = {rtype: counter.get(rtype, 0) for rtype in reduction_types}

    # Optionally unpack into variables


    return counts  # Or return locals() if you want all variables


if __name__ == "__main__":
    main()
