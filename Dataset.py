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
        "mushroom": 73,
        #"air_quality": 360, # 0 targets :/
        "estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition": 544,
        "spambase": 94,
        "seoul_bike_sharing_demand": 560,
        "optical_recognition_of_handwritten_digits": 80,
        "ai4i_2020_predictive_maintenance_dataset": 601, # 6 features for 6 targets...
        #"apartment_for_rent_classified": 555, # 0 targets :/
        "communities_and_crime": 183,
        "productivity_prediction_of_garment_employees": 597,
        "drug_consumption_quantified": 373, # 12 features for 19 targets...
        "myocardial_infarction_complications": 579,
        "parkinsons_telemonitoring": 189,
        "aids_clinical_trials_group_study_175": 890,
        "iranian_churn": 563,
        "taiwanese_bankruptcy_prediction": 572,
        "room_occupancy_estimation": 864,
        "infrared_thermography_temperature": 925,
        "support2": 880,
        "solar_flare": 89,
        "image_segmentation": 50,
        #"facebook_live_sellers_in_thailand": 488, # 0 targets :/
        "website_phishing": 379,
        "steel_plates_faults": 198,
        #"pedestrians_in_traffic": 536, # 0 targets :/
        "hepatitis_c_virus_hcv_for_egyptian_patients": 503,
        "communities_and_crime_unnormalized": 211,
        "statlog_landsat_satellite": 146,
        "mice_protein_expression": 342,
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
        self.variables = None

        self.n_instances = None
        self.n_features = None
        self.n_targets = None

        self.load()

    def load(self):
        dataset = fetch_ucirepo(id=self.id)
        self.X = dataset.data.features
        self.y = dataset.data.targets
        self.variables = dataset.variables

        self.n_instances = self.X.shape[0]
        self.n_features = len(self.variables[self.variables['role'] == 'Feature'])
        self.n_targets = len(self.variables[self.variables['role'] == 'Target'])

        if False and self.n_targets == 0:
            print(dataset.data.targets, flush=True)
            self.y.to_csv(f"{self.name}_targets.csv", index=False)

        print(f"    Dataset {dataset.metadata['name']}:\n"
              f"     - {self.n_instances} instances\n"
              f"     - {self.n_features} features\n"
              f"     - {self.n_targets} targets\n")

    @classmethod
    def load_n_datasets(cls, n=-1):
        dataset_ids = list(cls.DATASETS_IDS.items())[:n]
        datasets = []
        for i, (name, id) in enumerate(dataset_ids):
            print(f"Loading dataset {i+1}/{len(dataset_ids)}: {name} (ID: {id})")
            datasets.append(Dataset(id, name))
        return datasets

    @classmethod
    def load_sample(cls, id=186):
        return Dataset(id)
