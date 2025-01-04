from ucimlrepo import fetch_ucirepo
import pandas as pd


class Dataset:

    # First 30 datasets from UCI ML Repository using the following filters:
    # - 1K to 10K instances
    # - 10 to 4K features
    # - Available for python import
    DATASETS_IDS = {
        "wine_quality": 186,
        "predict_students_dropout_and_academic_success": 697,
        #"mushroom": 73, # missing values :/
        #"air_quality": 360, # 0 targets :/
        "estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition": 544,
        "spambase": 94,
        "seoul_bike_sharing_demand": 560,
        "optical_recognition_of_handwritten_digits": 80,
        #"ai4i_2020_predictive_maintenance_dataset": 601, # 6 features for 6 targets...
        #"apartment_for_rent_classified": 555, # 0 targets :/
        #"communities_and_crime": 183, # missing values :/
        #"productivity_prediction_of_garment_employees": 597, # missing values :/
        #"drug_consumption_quantified": 373, # 12 features for 19 targets...
        #"myocardial_infarction_complications": 579, #missing values :/
        "parkinsons_telemonitoring": 189,
        "aids_clinical_trials_group_study_175": 890,
        "iranian_churn": 563,
        "taiwanese_bankruptcy_prediction": 572,
        "room_occupancy_estimation": 864,
        "infrared_thermography_temperature": 925,
        #"support2": 880, # missing values :/
        "solar_flare": 89,
        "image_segmentation": 50,
        #"facebook_live_sellers_in_thailand": 488, # 0 targets :/
        "website_phishing": 379,
        "steel_plates_faults": 198,
        #"pedestrians_in_traffic": 536, # 0 targets :/
        "hepatitis_c_virus_hcv_for_egyptian_patients": 503,
        #"communities_and_crime_unnormalized": 211, # missing values :/ and same as communities_and_crime id 183
        "statlog_landsat_satellite": 146,
        #"mice_protein_expression": 342, # missing values :/
        "diabetic_retinopathy_debrecen": 329,
        "isolet": 54,
        "chess_king_rook_vs_king_pawn": 22,
        "waveform_database_generator_version_1": 107,
        "page_blocks_classification": 78,
        "musk_version_2": 75,
        "molecular_biology_splice_junction_gene_sequences": 69,
        "statlog_image_segmentation": 147
    }

    def __init__(self, id, name=None):
        self.id = id
        self.name = name or f"dataset_{id}"

        self.X = None
        self.y = None
        self.metadata = None
        self.variables = None

        self.n_instances = None
        self.n_features = None
        self.n_targets = None
        self.has_missing_values = None

        self.mappings = {}

        self.load()
        self.set_mappings()
        self.encode()

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

        self.n_instances = self.X.shape[0]
        self.n_features = len(self.variables[self.variables['role'] == 'Feature'])
        self.n_targets = len(self.variables[self.variables['role'] == 'Target'])
        self.has_missing_values = self.metadata['has_missing_values']

        print(dataset)

    def set_mappings(self):
        def yes_no(column_name):
            self.mappings[column_name] = {"yes": 1.0, "no": 0.0}

        def freq(column_name):
            self.mappings[column_name] = {"no": 0.0, "Sometimes": 1.0, "Frequently": 2.0, "Always": 3.0}

        match self.name:
            case "chess_king_rook_vs_king_pawn":
                self.mappings["wtoeg"] = {"win": 1.0, "nowin": 0.0}

            case "estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition":
                yes_no("family_history_with_overweight")
                yes_no("FAVC")
                freq("CAEC")
                freq("CALC")
                self.mappings["NObeyesdad"] = {
                    "Insufficient_Weight": 0.0,
                    "Normal_Weight": 1.0,
                    "Overweight_Level_I": 2.0,
                    "Overweight_Level_II": 3.0,
                    "Obesity_Type_I": 4.0,
                    "Obesity_Type_II": 5.0,
                    "Obesity_Type_III": 6.0
                }

            case "infrared_thermography_temperature":
                self.mappings["Age"] = {
                    "18-20": 0,
                    "21-30": 1,
                    "31-40": 2,
                    "41-50": 3,
                    "51-60": 4,
                    ">60": 5
                }

            case "molecular_biology_splice_junction_gene_sequences":
                for column_number in range(1, 60):  # we want exactly the same encoding for every base
                    self.mappings[f"Base{column_number}"] = {'A': 0., 'T': 1., 'C': 2., 'G': 3.}





    def encode(self):
        self.encode_df(self.X, self.mappings)
        self.encode_df(self.y, self.mappings)

    @staticmethod
    def encode_df(df, mappings):
        def auto_encode_column(column):  # auto-encode string column to float (1.0, 2.0, 3.0, ...)
            unique_values = column.unique()
            encoding_map = {value: float(idx) for idx, value in enumerate(unique_values)}
            return column.map(encoding_map)

        for column in df.columns:
            if df[column].dtype == object:  # (string)
                if column in mappings:  # special encoding
                    try:
                        df[column] = df[column].map(mappings[column])
                    except KeyError:  # we missed a value in the mapping
                        raise ValueError(f"Unsupported values for column '{column}' in df: {df[column].unique()}")
                df[column] = auto_encode_column(df[column])  # general encoding
            elif df[column].dtype == int:  # also changing integers to floats
                df[column] = df[column].astype(float)
            elif df[column].dtype != float:
                raise ValueError(f"Unsupported dtype for column '{column}' in df: {df[column].dtype}")

    @classmethod
    def load_datasets(cls, n=-1):
        dataset_ids = list(cls.DATASETS_IDS.items())[:n]
        datasets = []
        for i, (name, id) in enumerate(dataset_ids):
            print(f"Loading dataset {i + 1}/{len(dataset_ids)}: {name} (ID: {id})")
            datasets.append(Dataset(id, name))
        return datasets

    @classmethod
    def load_sample(cls, name="wine_quality"):
        return Dataset(cls.DATASETS_IDS[name], name)

