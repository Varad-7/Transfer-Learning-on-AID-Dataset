import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.visualize import plot_corruption_results

def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "scenario_4")
    json_path = os.path.join(output_dir, "s4_results.json")
    
    if not os.path.exists(json_path):
        print("s4_results.json not found!")
        return
        
    with open(json_path, "r") as f:
        all_results = json.load(f)
        
    print(f"Loaded {len(all_results)} results from JSON.")
                 
    plot_path = os.path.join(output_dir, "s4_corruption_results.png")
    plot_corruption_results(all_results, plot_path)
    print(f"Successfully re-generated {plot_path} with all models!")

if __name__ == "__main__":
    main()
