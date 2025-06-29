import gurobipy as gp
import os

INPUT_FILE = "Test_instances/0020.mps"
OUTPUT_DIR = "LP_models_of_original_instances"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "instance0020.lp")

def convert_mps_to_lp(input_path, output_path):
    model = gp.read(input_path)
    model.write(output_path)
    print(f"Converted: {input_path} → {output_path}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    convert_mps_to_lp(INPUT_FILE, OUTPUT_FILE)
