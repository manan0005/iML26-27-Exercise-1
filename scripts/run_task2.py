"""Task 2: Meta-features and interpretability."""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from iml_interpretability.metric import InterpretabilityComplexityScore

def main():
    # 1. Load three datasets with varying meta-features
    datasets = [
        ("Iris", load_iris(return_X_y=True)),
        ("Wine", load_wine(return_X_y=True)),
        ("Breast Cancer", load_breast_cancer(return_X_y=True))
    ]
    
    metric = InterpretabilityComplexityScore()
    results = []
    
    for name, (X, y) in datasets:
        # Calculate meta-features
        n_samples, n_features = X.shape
        majority_class_ratio = np.max(np.bincount(y)) / len(y)
        
        # Train an unrestricted Decision Tree to see how the dataset forces it to grow
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)
        
        # We pass None for dataset because our ICS metric only looks at the model structure
        ics_score = metric.score(clf, None) 
        
        results.append({
            "Dataset": name,
            "Samples": n_samples,
            "Features": n_features,
            "Majority Class %": round(majority_class_ratio * 100, 1),
            "Unrestricted Tree ICS": ics_score
        })
        
    df = pd.DataFrame(results)
    print("--- Task 2: Meta-Features vs. Interpretability ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()