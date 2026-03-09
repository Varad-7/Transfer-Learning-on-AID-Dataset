import os
import sys
import json
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.visualize import plot_few_shot_comparison

def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "scenario_3")
    json_path = os.path.join(output_dir, "s3_results.json")
    
    if not os.path.exists(json_path):
        print("s3_results.json not found!")
        return
        
    with open(json_path, "r") as f:
        all_results = json.load(f)
        
    print(f"Loaded {len(all_results)} results from JSON.")
    
    plot_data = [{"model_name": r["model_name"], "fraction": r["fraction"], "val_acc": r["val_acc"]}
                 for r in all_results]
                 
    plot_path = os.path.join(output_dir, "s3_few_shot_comparison.png")
    plot_few_shot_comparison(plot_data, plot_path)
    print(f"Successfully re-generated {plot_path} with all models!")

if __name__ == "__main__":
    main()
