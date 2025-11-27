"""simulation_ml.py
Data-driven ML simulation prototype for Workshop #4.

Usage:
    python simulation_ml.py --n_samples 8000 --output results_ml.csv

This script generates a synthetic clickstream (if no real CSV is provided),
builds a TF-IDF + temporal feature pipeline, trains a RandomForestClassifier,
and writes a summary CSV with Top-5 and MAP@5 metrics.

To run on real data: provide a CSV with columns ['user','sku','category','query','query_time','click_time']
and pass --input path/to/file.csv
"""

import argparse
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import top_k_accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

np.random.seed(42)

def generate_synthetic_clickstream_fast(n_samples=8000, n_users=2000, n_skus=800, n_queries_vocab=1500, n_categories=20):
    users = np.array([f"user_{i}" for i in range(n_users)])
    skus = np.array([f"sku_{i}" for i in range(n_skus)])
    categories = np.array([f"cat_{i}" for i in range(n_categories)])
    weights = np.ones(n_categories)
    weights[0:2] = 7
    weights = weights / weights.sum()
    words = np.array([f"term{i}" for i in range(n_queries_vocab)])

    user_idx = np.random.randint(0, n_users, size=n_samples)
    sku_idx = np.random.randint(0, n_skus, size=n_samples)
    cat_idx = np.random.choice(np.arange(n_categories), size=n_samples, p=weights)
    lengths = np.random.choice([1,2,3], size=n_samples, p=[0.6,0.3,0.1])
    queries = []
    for L in lengths:
        queries.append(" ".join(np.random.choice(words, size=L)))
    queries = np.array(queries)

    base_time = pd.Timestamp("2023-01-01")
    offsets = np.random.exponential(scale=3600*24, size=n_samples).astype(int)
    qtimes = base_time + pd.to_timedelta(offsets, unit='s')
    click_offsets = np.random.randint(1,300,size=n_samples)
    click_times = qtimes + pd.to_timedelta(click_offsets, unit='s')
    df = pd.DataFrame({
        "user": users[user_idx],
        "sku": skus[sku_idx],
        "category": categories[cat_idx],
        "query": queries,
        "query_time": qtimes,
        "click_time": click_times
    })
    return df

def map_at_k(y_true, probs, k=5):
    n = probs.shape[0]
    ap_sum = 0.0
    for i in range(n):
        topk = np.argsort(probs[i])[::-1][:k]
        true = y_true[i]
        score = 0.0
        hits = 0
        for rank, idx in enumerate(topk, start=1):
            if idx == true:
                hits += 1
                score += hits / rank
        if hits > 0:
            ap_sum += score / hits
    return ap_sum / n

def run(args):
    if args.input:
        df = pd.read_csv(args.input, parse_dates=['query_time','click_time'])
    else:
        df = generate_synthetic_clickstream_fast(n_samples=args.n_samples)
    df['query_norm'] = df['query'].str.lower().fillna("")
    le = LabelEncoder()
    y = le.fit_transform(df['category'])
    tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1,2))
    X_tfidf = tfidf.fit_transform(df['query_norm'])
    df['delta_sec'] = (df['click_time'] - df['query_time']).dt.total_seconds().astype(int).clip(0,300)
    X_time = df[['delta_sec']].values
    X = hstack([X_tfidf, X_time])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
    t0 = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - t0
    probs = rf.predict_proba(X_test)
    top5_acc = top_k_accuracy_score(y_test, probs, k=5, labels=np.arange(len(le.classes_)))
    map5 = map_at_k(y_test, probs, k=5)
    result = {
        'n_samples': [df.shape[0]],
        'top5_acc':[float(top5_acc)],
        'map5':[float(map5)],
        'train_time_s':[float(train_time)]
    }
    out = pd.DataFrame(result)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))
    print(f"Saved results to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None, help='Path to real CSV with required columns')
    parser.add_argument('--n_samples', type=int, default=8000)
    parser.add_argument('--output', type=str, default='results_ml.csv')
    args = parser.parse_args()
    run(args)
