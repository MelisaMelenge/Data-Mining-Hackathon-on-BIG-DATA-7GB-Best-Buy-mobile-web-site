Workshop 4 - Simulation Implementation
Files included:
- simulation_ml.py : ML simulation (TF-IDF + RandomForest). Generates synthetic data if no input CSV provided.
- simulation_ca.py : Cellular Automata event-based simulation.
- requirements.txt : Python dependencies.
Usage examples:
1) Run ML simulation with synthetic data:
   python simulation_ml.py --n_samples 8000 --output results_ml.csv
2) Run CA simulation:
   python simulation_ca.py --steps 100 --rows 50 --cols 50 --output ca_counts.csv
To run on real data:
 - Provide a CSV with columns: user, sku, category, query, query_time, click_time
 - Example: python simulation_ml.py --input train_sample.csv --output results_ml.csv
