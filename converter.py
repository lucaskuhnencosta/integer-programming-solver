import os
import gurobipy as gp

input_dir = "LP_models_of_original_instances"
output_dir = "MercadoLivre_instances"

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.endswith(".lp"):
        lp_path = os.path.join(input_dir, fname)
        mps_path = os.path.join(output_dir, fname.replace(".lp", ".mps"))
        model = gp.read(lp_path)
        model.write(mps_path)
        print(f"Converted {fname} to {os.path.basename(mps_path)}")
