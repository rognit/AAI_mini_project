import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import rand_score

from dataset_config import DATASETS_IDS
from Dataset import Dataset

def process_dataset(name, dataset_id):
    try:
        # Load the dataset
        dataset = Dataset(id=dataset_id, name=name)
        X = dataset.X
        y = dataset.y
        y = y.iloc[:, 0].astype('category').cat.codes

        # Train DBSCAN model 
        model = DBSCAN(eps=0.1, min_samples=100)

        # Fit and predict the labels
        y_pred = model.fit_predict(X)

        score = rand_score(y, y_pred)
        return {
            "dataset_id": dataset_id,
            "name": name,
            "rand_score": score
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
    print(np.mean(results_df['rand_score']))
    return results_df

if __name__ == "__main__":
    results_df = main()
