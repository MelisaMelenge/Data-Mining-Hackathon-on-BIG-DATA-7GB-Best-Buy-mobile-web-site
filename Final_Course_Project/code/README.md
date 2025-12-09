# BestBuy Product Recommendation 

This directory contains all the components needed to run a Random Forest-based product recommendation pipeline. The system processes user search queries and predicts which BestBuy products a mobile web visitor will be most interested in based on their behavior.

## Directory Structure

```
├── dataset/             # Folder containing all required input data files
│   ├── train.csv       # Training data with user queries and product interactions
│   ├── test.csv        # Test data for generating predictions
│   ├── new_train.csv   # Preprocessed training data (generated)
│   └── new_test.csv    # Preprocessed test data (generated)
├── src/                 # Source code directory
│   ├── main.py         # Main script for generating predictions
│   ├── ml_model.py     # Random Forest ML model implementation with API interface
│   ├── data_processing.py  # Data preprocessing and query normalization
│   ├── util.py         # Utility functions for data handling
│   └── constants.py    # Configuration constants
├── output/              # Output directory (auto-created)
│   ├── random_forest_model.pkl  # Trained model file
│   └── prediction.csv  # Final predictions output
├── requirements.txt     # Python dependencies for the project
```

## Prerequisites

- Python 3.8 or higher

Install dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Download NLTK data (required for text preprocessing):

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Usage

1. **Prepare the data**: Place `train.csv` and `test.csv` in the `maindata/` folder.

2. **Preprocess the data**:
   ```bash
   python src/data_processing.py
   ```

3. **Train the Random Forest model**:
   ```bash
   python src/ml_model.py
   ```

4. **Generate predictions**:
   ```bash
   python src/main.py
   ```

After execution, inspect `output/prediction.csv` for the model's predictions.

## Output

**prediction.csv**: CSV file containing the top 5 predicted product SKUs for each test query, space-separated. If no prediction is available for a query, it outputs "0".

Example output format:
```
sku
2125233 2009324 1517163
3108172 9755322 1534115
0
```

## Model Details

The system uses a **Random Forest classifier** with the following features:
- Query-based features (length, word frequencies, bigrams)
- Temporal features (click time)
- Category-specific models for improved accuracy
- Feature engineering with NLTK lemmatization




