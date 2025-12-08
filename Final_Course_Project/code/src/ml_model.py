"""
Machine Learning Model Module - Random Forest Implementation
This module provides a Random Forest classifier as the ML model component
required for the competition, with an API/connector interface.
"""

import pickle
import os
from collections import defaultdict
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from util import readfile, get_words
from constants import new_train_file, new_test_file, out_buffer_path

class ProductRecommenderML:
    """
    ML Model API/Connector for product recommendations using Random Forest.
    This class serves as the interface between the data and the ML model.
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.label_encoder = LabelEncoder()
        self.category_models = {}
        self.feature_names = []
        
    def extract_features(self, query, category, click_time, word_freq, bigram_freq):
        """
        Feature engineering: Extract features from query and context.
        This transforms raw data into ML-ready features.
        """
        features = []
        words = get_words(query)
        
        # Query-based features
        features.append(len(words))  # Query length
        features.append(len(query))  # Character count
        
        # Word frequency features (top 50 most common words)
        for word in self.top_words[:50]:
            features.append(1 if word in words else 0)
        
        # Bigram features (top 30 most common bigrams)
        bigrams = self._get_bigrams(words)
        for bigram in self.top_bigrams[:30]:
            features.append(1 if bigram in bigrams else 0)
        
        # Time feature
        features.append(int(click_time))
        
        # Word frequency in category
        avg_freq = np.mean([word_freq.get(w, 0) for w in words]) if words else 0
        features.append(avg_freq)
        
        # Bigram frequency in category
        avg_bigram_freq = np.mean([bigram_freq.get(b, 0) for b in bigrams]) if bigrams else 0
        features.append(avg_bigram_freq)
        
        return features
    
    def _get_bigrams(self, words):
        """Generate bigrams from word list"""
        if len(words) < 2:
            return []
        words = sorted(words)
        return ['_'.join([words[i], words[j]]) for i in range(len(words)) for j in range(i+1, len(words))]
    
    def train(self, train_file):
        """
        Train the Random Forest model on the training data.
        API Method: Loads and processes data, then trains the model.
        """
        print("Training Random Forest model...")
        
        # Load training data
        reader = readfile(train_file)
        data_by_category = defaultdict(list)
        
        # Collect word and bigram frequencies
        word_freq_global = defaultdict(int)
        bigram_freq_global = defaultdict(int)
        
        for (user, sku, category, query, click_time) in reader:
            words = get_words(query)
            for word in words:
                word_freq_global[word] += 1
            bigrams = self._get_bigrams(words)
            for bigram in bigrams:
                bigram_freq_global[bigram] += 1
            data_by_category[category].append((user, sku, query, click_time))
        
        # Get top features
        self.top_words = sorted(word_freq_global.items(), key=lambda x: x[1], reverse=True)[:50]
        self.top_words = [w[0] for w in self.top_words]
        self.top_bigrams = sorted(bigram_freq_global.items(), key=lambda x: x[1], reverse=True)[:30]
        self.top_bigrams = [b[0] for b in self.top_bigrams]
        
        # Train a model per category
        trained_categories = 0
        for category, records in data_by_category.items():
            if len(records) < 10:  # Skip categories with too few samples
                continue
            
            # Build word/bigram frequency per category
            word_freq = defaultdict(int)
            bigram_freq = defaultdict(int)
            for (_, _, query, _) in records:
                words = get_words(query)
                for word in words:
                    word_freq[word] += 1
                bigrams = self._get_bigrams(words)
                for bigram in bigrams:
                    bigram_freq[bigram] += 1
            
            # Prepare features and labels
            X = []
            y = []
            for (_, sku, query, click_time) in records:
                features = self.extract_features(query, category, click_time, word_freq, bigram_freq)
                X.append(features)
                y.append(sku)
            
            # Train model for this category
            if len(set(y)) >= 2:  # Need at least 2 different products
                model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=15,
                    min_samples_split=3,
                    random_state=42,
                    n_jobs=-1
                )
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
                model.fit(X, y_encoded)
                self.category_models[category] = (model, label_encoder, word_freq, bigram_freq)
                trained_categories += 1
        
        print(f"Trained models for {trained_categories} categories")
        return self
    
    def predict(self, query, category, click_time, top_k=5):
        """
        API Method: Get top-k product predictions for a given query.
        This is the inference interface.
        """
        if category not in self.category_models:
            return []
        
        model, label_encoder, word_freq, bigram_freq = self.category_models[category]
        
        # Extract features
        features = self.extract_features(query, category, click_time, word_freq, bigram_freq)
        X = np.array([features])
        
        # Get predictions with probabilities
        try:
            proba = model.predict_proba(X)[0]
            top_indices = np.argsort(proba)[::-1][:top_k]
            predictions = [label_encoder.classes_[idx] for idx in top_indices]
            return predictions
        except:
            return []
    
    def save_model(self, path):
        """Save the trained model to disk"""
        with open(path, 'wb') as f:
            pickle.dump({
                'category_models': self.category_models,
                'top_words': self.top_words,
                'top_bigrams': self.top_bigrams
            }, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.category_models = data['category_models']
            self.top_words = data['top_words']
            self.top_bigrams = data['top_bigrams']
        print(f"Model loaded from {path}")
        return self


def train_ml_model():
    """
    Connector function: Trains the ML model and saves it.
    This serves as the main API entry point for training.
    """
    model = ProductRecommenderML()
    model.train(new_train_file)
    
    model_path = os.path.join(out_buffer_path, 'random_forest_model.pkl')
    model.save_model(model_path)
    return model


def load_ml_model():
    """
    Connector function: Loads a pre-trained ML model.
    This serves as the main API entry point for inference.
    """
    model_path = os.path.join(out_buffer_path, 'random_forest_model.pkl')
    if os.path.exists(model_path):
        model = ProductRecommenderML()
        model.load_model(model_path)
        return model
    else:
        print("No saved model found. Training new model...")
        return train_ml_model()


if __name__ == '__main__':
    # Train and save the model
    print("Training Random Forest ML Model...")
    model = train_ml_model()
    print("Done! Model trained and saved.")
