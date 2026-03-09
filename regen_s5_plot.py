import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.visualize import plot_layerwise_accuracy, plot_feature_norms

def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "scenario_5")
    json_path = os.path.join(output_dir, "s5_results.json")
    
    if not os.path.exists(json_path):
        print("s5_results.json not found!")
        return
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    all_results = data.get("accuracy_results", [])
    all_norm_stats = data.get("norm_stats", [])
        
    print(f"Loaded {len(all_results)} accuracy results and {len(all_norm_stats)} norm stats from JSON.")
                 
    # Plot accuracy vs depth
    plot_path_acc = os.path.join(output_dir, "s5_acc_vs_depth.png")
    plot_layerwise_accuracy(all_results, plot_path_acc)
    print(f"Successfully re-generated {plot_path_acc} with all models!")
    
    # Plot feature norms
    plot_path_norms = os.path.join(output_dir, "s5_feature_norms.png")
    plot_feature_norms(all_norm_stats, plot_path_norms)
    print(f"Successfully re-generated {plot_path_norms} with all models!")

if __name__ == "__main__":
    main()
