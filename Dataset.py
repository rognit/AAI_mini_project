import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from dataset_config import DATASETS_IDS, get_mappings
from Encoder import Encoder

class Dataset:
    def __init__(self, id, name=None, norm='standard', pca=.95):
        self.id = id
        self.name = name or f"dataset_{id}"
        self.X = None
        self.y = None

        self.X_raw = None
        self.y_raw = None
        self.metadata = None
        self.variables = None

        self.n_instances = None
        self.n_raw_features = None
        self.n_raw_targets = None
        self.has_missing_values = None

        self.pca_n_components = None
        self.pca_variance_ratio = None
        self.pca_cumulative_variance = None

        self.mappings = get_mappings(self.name)

        self._load()
        self._encode()
        self._normalize(norm)
        self._apply_pca(pca)

        self.n_features = self.X.shape[1]
        self.n_targets = self.y.shape[1]

        print(self, flush=True)

    def _load(self):
        dataset = fetch_ucirepo(id=self.id)

        self.X = dataset.data.features
        self.y = dataset.data.targets
        self.metadata = dataset.metadata
        self.variables = dataset.variables

        self.X_raw = self.X.copy()
        self.y_raw = self.y.copy()

        self.n_instances = self.X.shape[0]
        self.n_raw_features = len(self.variables[self.variables['role'] == 'Feature'])
        self.n_raw_targets = len(self.variables[self.variables['role'] == 'Target'])
        self.has_missing_values = self.metadata['has_missing_values']

    def _encode(self):
        if 'Date' in self.X.columns:  # encoding + enrichment
            self.X = Encoder.encode_enrich_date(self.X)
        if 'Time' in self.X.columns:  # encoding + enrichment
            self.X = Encoder.encode_enrich_time(self.X)

        Encoder.encode_df(self.X, self.mappings)
        Encoder.encode_df(self.y, self.mappings)

    def _normalize(self, method):
        if method not in ['standard', 'minmax']:
            raise ValueError("Method must be either 'standard' or 'minmax'")

        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        self.X = pd.DataFrame(scaler.fit_transform(self.X), columns=self.X.columns, index=self.X.index)
        self.y = pd.DataFrame(scaler.fit_transform(self.y), columns=self.y.columns, index=self.y.index)

    def _apply_pca(self, variance_threshold):
        # Initialize PCA with maximum possible components, fit it to get variance ratios and compute the variance ratios
        pca = PCA()
        pca.fit(self.X)

        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

        # Number of components needed to meet variance threshold
        self.pca_n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
        self.pca_variance_ratio = pca.explained_variance_ratio_
        self.pca_cumulative_variance = cumulative_variance_ratio

        # Apply PCA with selected number of components
        pca = PCA(n_components=self.pca_n_components)

        column_names = [f'PC{i + 1}' for i in range(self.pca_n_components)]  # Convert to DataFrame with meaningful column names
        self.X = pd.DataFrame(
            pca.fit_transform(self.X),
            columns=column_names,
            index=self.X.index
        )

        print(f"  Applied PCA transformation:\n"
              f"   - Reduced features from {len(self.pca_variance_ratio)} to {self.pca_n_components} "
              f"({100 * (len(self.pca_variance_ratio) - self.pca_n_components) / len(self.pca_variance_ratio):.0f}% "
              f"reduction)\n"
              f"   - Preserved {cumulative_variance_ratio[self.pca_n_components - 1] * 100:.2f}% of variance", flush=True)

    def __str__(self):
        return (f"  Final dataset {self.metadata['name']}:\n"
                f"   - {self.n_instances} instances\n"
                f"   - {self.n_features} features\n"
                f"   - {self.n_targets} targets\n"
                f"   - has missing values: {self.has_missing_values}\n")

    def visualize(self):
        self.X.to_csv(f"{self.name}_X.csv", index=False)
        self.X_raw.to_csv(f"{self.name}_X_raw.csv", index=False)
        self.y.to_csv(f"{self.name}_y.csv", index=False)
        self.y_raw.to_csv(f"{self.name}_y_raw.csv", index=False)

    @classmethod
    def load_datasets(cls, n=-1):
        dataset_ids = list(DATASETS_IDS.items())[:n]
        datasets = []
        for i, (name, id) in enumerate(dataset_ids):
            print(f"Loading dataset {i + 1}/{len(dataset_ids)}: {name} (ID: {id})", flush=True)
            datasets.append(Dataset(id, name))
        return datasets

    @classmethod
    def load_sample(cls, name="wine_quality"):
        return Dataset(DATASETS_IDS[name], name)
