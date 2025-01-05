import pandas as pd
import numpy as np

class Encoder:
    @staticmethod
    def encode_enrich_date(df):
        """Transform 'Date' into 'day_of_month', 'day_of_week', 'month', 'days_since_start'."""
        try:
            dates = pd.to_datetime(df['Date'], format='%Y/%m/%d')
        except ValueError:
            try:
                dates = pd.to_datetime(df['Date'], format='%d/%m/%Y')
            except ValueError:
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
        """Transform 'Time' into 'hour', 'minute', 'hour_sin', 'hour_cos', etc."""
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
        """Auto-encode categorical and numerical features in the DataFrame."""
        def auto_encode_column(column, alpha=False):
            unique_values = sorted(column.unique()) if alpha else column.unique()
            encoding_map = {value: float(idx) for idx, value in enumerate(unique_values)}
            return column.map(encoding_map)

        for column in df.columns:
            if df[column].dtype == object:  # String type
                if column in mappings:  # Custom mappings
                    try:
                        df[column] = df[column].map(mappings[column])
                    except KeyError:
                        raise ValueError(f"Unsupported values for column '{column}' in df: {df[column].unique()}")
                else:
                    df[column] = auto_encode_column(df[column])  # General encoding
            elif df[column].dtype == np.int64:  # Convert integers to floats
                df[column] = df[column].astype(float)
            elif df[column].dtype != float:
                raise ValueError(f"Unsupported dtype for column '{column}' in df: {df[column].dtype}")
