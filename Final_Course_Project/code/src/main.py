"""
Main prediction module using Random Forest ML model.
Simplified version for competition - Random Forest only.
"""

import os
import sys
_root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from util import readfile, writefile
from constants import new_test_file, out_file
from ml_model import load_ml_model

def make_predictions():
    """
    Generate predictions using Random Forest ML model.
    Loads trained model and predicts top 5 products for each test query.
    """
    # Load or train the Random Forest model
    print("Loading Random Forest ML model...")
    model = load_ml_model()
    
    # Read test data and make predictions
    reader = readfile(new_test_file)
    writer = writefile(out_file)
    
    print("Generating predictions...")
    for (user, category, query, click_time) in reader:
        # Get predictions from Random Forest
        predictions = model.predict(query, category, click_time, top_k=5)
        
        if predictions:
            writer.writerow([" ".join(predictions)])
        else:
            # Fallback: return placeholder if no prediction available
            writer.writerow(["0"])
    
    print("Predictions saved to:", out_file)

def main():
    make_predictions()
    print('Done')

if __name__ == '__main__':
    main()
