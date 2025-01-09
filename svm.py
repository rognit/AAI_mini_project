import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from dataset_config import DATASETS_IDS
from Dataset import Dataset

def process_dataset(name, dataset_id):
    try:
        # Load the dataset
        dataset = Dataset(id=dataset_id, name=name)
        X = dataset.X
        y = dataset.y

        # Allows you to have just one target, and for it to be a category.
        y = y.iloc[:, 0].astype('category').cat.codes

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = SVC()
        model.fit(X_train, y_train)

        # Test model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return {
            "dataset_id": dataset_id,
            "name": name,
            "accuracy": accuracy
        }
    except Exception as e:
        return {"dataset_id": dataset_id, "error": f"Processing error: {str(e)}"}

def main():
    results = []
    for name, dataset_id in DATASETS_IDS.items():
        print(f"Start dataset: {name} (ID: {dataset_id})")
        result = process_dataset(name, dataset_id)
        results.append(result)
        print(f"Result: {result}\n")
    
    results_df = pd.DataFrame(results)
    print("\n=== Final Result ===\n")
    print(results_df)
    print(np.mean(results_df['accuracy']))
    return results_df

if __name__ == "__main__":
    results_df = main()
