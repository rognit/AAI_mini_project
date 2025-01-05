from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from dataset_config import DATASETS_IDS, get_mappings

class Dataset:
    def __init__(self, id, name=None):
        self.id = id
        self.name = name or f"dataset_{id}"

        self.X_raw = None
        self.y_raw = None
        self.metadata = None
        self.variables = None

        self.n_instances = None
        self.n_raw_features = None
        self.n_raw_targets = None
        self.has_missing_values = None

        self.mappings = {}
        self.X = None
        self.y = None

        self.load()
        self.mappings = get_mappings(self.name)
        self.encode()

        self.n_features = len(self.X.columns)
        self.n_targets = len(self.y.columns)

        print(self)

    def __str__(self):
        return (f"    Dataset {self.metadata['name']}:\n"
                f"     - {self.n_instances} instances\n"
                f"     - {self.n_features} features\n"
                f"     - {self.n_targets} targets\n"
                f"     - has missing values: {self.has_missing_values}\n")

    def load(self):
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

    @staticmethod
    def encode_enrich_date(df):
        """
        transform 'Date' into 'day_of_month, day_of_week, month, days_since_start'
        """

        try:
            dates = pd.to_datetime(df['Date'], format='%Y/%m/%d')
        except ValueError:
            try:
                dates = pd.to_datetime(df['Date'], format='%d/%m/%Y')
            except ValueError: # If both fail, let pandas try to figure it out
                dates = pd.to_datetime(df['Date'])

        df['day_of_month'] = dates.dt.day / 31.  # [0,1]
        df['day_of_week'] = dates.dt.dayofweek / 6.  # [0,1]
        df['month'] = (dates.dt.month - 1) / 11.  # [0,1]

        min_date = dates.min()
        max_date = dates.max()
        days_diff = (max_date - min_date).days
        if days_diff == 0:  # Handle single day case
            df['days_since_start'] = 0.
        else:
            df['days_since_start'] = (dates - min_date).dt.total_seconds() / (24 * 60 * 60) / days_diff

        return df.drop('Date', axis=1)

    @staticmethod
    def encode_enrich_time(df):
        """
        transform 'Time' into 'hour, minute, hour_sin, hour_cos, minute_sin, minute_cos, minutes_since_midnight'
        """

        times = pd.to_datetime('2000-01-01 ' + df['Time'].astype(str))  # (adding a dummy date)

        df['hour'] = times.dt.hour / 23.  # [0,1]
        df['minute'] = times.dt.minute / 59.  # [0,1]

        df['hour_sin'] = np.sin(2 * np.pi * times.dt.hour / 24.)
        df['hour_cos'] = np.cos(2 * np.pi * times.dt.hour / 24.)
        df['minute_sin'] = np.sin(2 * np.pi * times.dt.minute / 60.)
        df['minute_cos'] = np.cos(2 * np.pi * times.dt.minute / 60.)

        minutes_since_midnight = times.dt.hour * 60. + times.dt.minute
        df['minutes_since_midnight'] = minutes_since_midnight / (24. * 60.)  # [0,1]

        return df.drop('Time', axis=1)

    @staticmethod
    def encode_df(df, mappings):
        def auto_encode_column(column, alpha=False):  # auto-encode string column to float (1.0, 2.0, 3.0, ...)
            unique_values = sorted(column.unique()) if alpha else column.unique()
            encoding_map = {value: float(idx) for idx, value in enumerate(unique_values)}
            return column.map(encoding_map)

        for column in df.columns:
            if df[column].dtype == object:  # (string)
                if column in mappings:  # special encoding
                    try:
                        df[column] = df[column].map(mappings[column])
                    except KeyError:  # means we missed a value in the manual mapping
                        raise ValueError(f"Unsupported values for column '{column}' in df: {df[column].unique()}")
                else:
                    df[column] = auto_encode_column(df[column])  # general encoding
            elif df[column].dtype == np.int64:  # also changing integers to floats
                df[column] = df[column].astype(float)
            elif df[column].dtype != float:
                raise ValueError(f"Unsupported dtype for column '{column}' in df: {df[column].dtype}")

    def encode(self):
        if 'Date' in self.X.columns:  # encoding + enrichment
            self.X = self.encode_enrich_date(self.X)
        if 'Time' in self.X.columns:  # encoding + enrichment
            self.X = self.encode_enrich_time(self.X)

        self.encode_df(self.X, self.mappings)
        self.encode_df(self.y, self.mappings)
        #self.X.to_csv(f"{self.name}_X.csv", index=False)
        #self.X_raw.to_csv(f"{self.name}_X_raw.csv", index=False)

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
            print(f"Loading dataset {i + 1}/{len(dataset_ids)}: {name} (ID: {id})")
            datasets.append(Dataset(id, name))
        return datasets

    @classmethod
    def load_sample(cls, name="wine_quality"):
        return Dataset(DATASETS_IDS[name], name)
