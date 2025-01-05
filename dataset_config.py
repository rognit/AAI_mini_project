# Configuration file for dataset IDs and mappings

# First 30 datasets from UCI ML Repository using the following filters:
# - 1K to 10K instances
# - 10 to 4K features
# - Available for python import
DATASETS_IDS = {
    "wine_quality": 186,
    "predict_students_dropout_and_academic_success": 697,
    # "mushroom": 73, # missing values :/
    # "air_quality": 360, # 0 targets :/
    "estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition": 544,
    "spambase": 94,
    "seoul_bike_sharing_demand": 560,
    "optical_recognition_of_handwritten_digits": 80,
    # "ai4i_2020_predictive_maintenance_dataset": 601, # 6 features for 6 targets...
    # "apartment_for_rent_classified": 555, # 0 targets :/
    # "communities_and_crime": 183, # missing values :/
    # "productivity_prediction_of_garment_employees": 597, # missing values :/
    # "drug_consumption_quantified": 373, # 12 features for 19 targets...
    # "myocardial_infarction_complications": 579, #missing values :/
    "parkinsons_telemonitoring": 189,
    "aids_clinical_trials_group_study_175": 890,
    "iranian_churn": 563,
    "taiwanese_bankruptcy_prediction": 572,
    "room_occupancy_estimation": 864,
    "infrared_thermography_temperature": 925,
    # "support2": 880, # missing values :/
    "solar_flare": 89,
    "image_segmentation": 50,
    # "facebook_live_sellers_in_thailand": 488, # 0 targets :/
    "website_phishing": 379,
    "steel_plates_faults": 198,
    # "pedestrians_in_traffic": 536, # 0 targets :/
    "hepatitis_c_virus_hcv_for_egyptian_patients": 503,
    # "communities_and_crime_unnormalized": 211, # missing values :/ and same as communities_and_crime id 183
    "statlog_landsat_satellite": 146,
    # "mice_protein_expression": 342, # missing values :/
    "diabetic_retinopathy_debrecen": 329,
    "isolet": 54,
    "chess_king_rook_vs_king_pawn": 22,
    "waveform_database_generator_version_1": 107,
    "page_blocks_classification": 78,
    "musk_version_2": 75,
    "molecular_biology_splice_junction_gene_sequences": 69,
    "statlog_image_segmentation": 147
}


def get_mappings(dataset_name):
    """Returns the encoding mappings for a specific dataset."""

    def yes_no():
        return {"No": 0., "no": 0., "Yes": 1., "yes": 1.}

    def freq():
        return {"no": 0., "Sometimes": 1., "Frequently": 2., "Always": 3.}

    mappings = {}

    match dataset_name:
        case "chess_king_rook_vs_king_pawn":
            mappings["wtoeg"] = {"nowin": 0., "win": 1.}

        case "estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition":
            mappings.update({
                "family_history_with_overweight": yes_no(),
                "FAVC": yes_no(),
                "CAEC": freq(),
                "CALC": freq(),
                "NObeyesdad": {
                    "Insufficient_Weight": 0.,
                    "Normal_Weight": 1.,
                    "Overweight_Level_I": 2.,
                    "Overweight_Level_II": 3.,
                    "Obesity_Type_I": 4.,
                    "Obesity_Type_II": 5.,
                    "Obesity_Type_III": 6.
                }
            })

        case "infrared_thermography_temperature":
            mappings["Age"] = {
                "18-20": 0.,
                "21-25": 1.,
                "21-30": 1.5,
                "26-30": 2.,
                "31-40": 3.,
                "41-50": 4.,
                "51-60": 5.,
                ">60": 6.
            }

        case "molecular_biology_splice_junction_gene_sequences":
            base_mapping = {"A": 0., "C": 1., "G": 2., "T": 3.}
            mappings.update({f"Base{i}": base_mapping.copy() for i in range(1, 60)})

        case "room_occupancy_estimation_features":
            mappings["Date"] = {}
            mappings["Time"] = {}

        case "seoul_bike_sharing_demand":
            mappings["Functioning Day"] = yes_no()

        case "solar_flare":
            mappings["modified Zurich class"] = {"B": 1., "C": 2., "D": 3., "E": 4., "F": 5., "H": 7.}

    return mappings