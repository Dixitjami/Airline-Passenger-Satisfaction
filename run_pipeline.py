#!/usr/bin/env python
"""
Airline Passenger Satisfaction - Full Pipeline Runner

Runs the complete ML pipeline:
1. EDA (01_EDA.py)
2. Data Cleaning (02_data_cleaning.py)
3. Model Training (train_baseline.py, train_decision_tree.py, train_random_forest.py, train_gradient_boosting.py)
4. Final Evaluation (evaluate_final.py)
5. Model Comparison (model_comparison.py)
"""

import subprocess
import sys
import os

def run_script(script_name, description, use_module=False):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    if use_module:
        # Run as module: python -m notebooks.script_name
        module_name = script_name.replace('.py', '').replace('/', '.')
        cmd = [sys.executable, '-m', f'notebooks.{module_name}']
    else:
        script_path = os.path.join("src" if script_name.startswith("train_") or script_name == "evaluate_final.py" else "notebooks", script_name)
        cmd = [sys.executable, script_path]
    
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    
    if result.returncode == 0:
        print(f"SUCCESS: {description}")
        if result.stdout:
            print(result.stdout[-1000:])  # Last 1000 chars
        return True
    else:
        print(f"FAILED: {description}")
        print(f"Error: {result.stderr[-2000:]}")
        return False


def main():
    print("Airline Passenger Satisfaction - Full Pipeline")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("dataset/train.csv"):
        print("ERROR: dataset/train.csv not found. Run from project root.")
        return 1
    
    scripts = [
        ("01_EDA.py", "EDA Analysis", True),
        ("02_data_cleaning.py", "Data Cleaning", True),
        ("train_baseline.py", "Logistic Regression Baseline", False),
        ("train_decision_tree.py", "Decision Tree", False),
        ("train_random_forest.py", "Random Forest", False),
        ("train_gradient_boosting.py", "Gradient Boosting", False),
        ("evaluate_final.py", "Final Test Evaluation", False),
        ("model_comparison.py", "Model Comparison", True),
    ]
    
    failed = []
    for script, desc, use_module in scripts:
        if not run_script(script, desc, use_module):
            failed.append(desc)
    
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    
    if failed:
        print(f"FAILED ({len(failed)}): {', '.join(failed)}")
        return 1
    else:
        print("ALL STEPS COMPLETED SUCCESSFULLY!")
        print("\nGenerated artifacts:")
        print("  - models/*.joblib (trained models)")
        print("  - reports/*.json (metrics)")
        print("  - reports/figures/*.png (visualizations)")
        print("  - logs/*.log (pipeline logs)")
        return 0


if __name__ == "__main__":
    sys.exit(main())