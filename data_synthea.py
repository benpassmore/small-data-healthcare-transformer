import pandas as pd

DATA_PATH = r"C:\Users\benpa\OneDrive\Desktop\UT MSDS\DSC AI in Healthcare\High-risk Project\project_data\Synthea\csv"

# Load Synthea data first
patients_df = pd.read_csv(f"{DATA_PATH}\\patients.csv")
conditions_df = pd.read_csv(f"{DATA_PATH}\\conditions.csv")
observations_df = pd.read_csv(f"{DATA_PATH}\\observations.csv")
dataset_end_date = pd.to_datetime(observations_df["DATE"]).max()
print("Dataset end date:", dataset_end_date)
medications_df = pd.read_csv(f"{DATA_PATH}\\medications.csv")

# Build one dictionary per Synthea patient
# Conditions all unique condition descriptions for each patient
conditions_by_patient = (
    conditions_df.groupby("PATIENT")["DESCRIPTION"]
    .apply(lambda x: sorted(set(x.dropna())))
    .to_dict())

# Medication all unique medication descriptions for each patient
medications_by_patient = (
    medications_df.groupby("PATIENT")["DESCRIPTION"]
    .apply(lambda x: sorted(set(x.dropna())))
    .to_dict())

# Observation names we want to keep
observation_names = {
    "Body mass index (BMI) [Ratio]": "bmi",
    "Body Weight": "weight",
    "Systolic Blood Pressure": "systolic_bp",
    "Diastolic Blood Pressure": "diastolic_bp",
    "Tobacco smoking status": "smoking",
    "Highest level of education": "education",
    "Employment status - current": "employment",
    "Housing status": "housing",
    "Has lack of transportation kept you from medical appointments  meetings  work  or from getting things needed for daily living": "transportation",
    "How often do you see or talk to people that you care about and feel close to [PRAPARE]": "social_connection"
}

# Keep only the observations we need
selected_observations = observations_df[observations_df["DESCRIPTION"].isin(observation_names)].copy()

# Convert dates and take patients most recent measurement
selected_observations["DATE"] = pd.to_datetime(selected_observations["DATE"])
latest_observations = (selected_observations.sort_values("DATE").drop_duplicates(subset=["PATIENT", "DESCRIPTION"], keep="last"))

# Convert observation descriptions into our simpler field names on the RHS of the: in observation_names
latest_observations["FIELD"] = latest_observations["DESCRIPTION"].map(observation_names)

# Create lookup patient ID 
observations_by_patient = {}

# creating a dict of dicts starting with builiding the patients
# this is becuase the observation data has one column
# with all the types of observaitons under one description column 
# instead of unique columns ughhh!
for _, row in latest_observations.iterrows():
    patient_id = row["PATIENT"]
    # if new patietn add them into the dict with a blank dict
    if patient_id not in observations_by_patient:
        observations_by_patient[patient_id] = {}
    # populate teh patient's personal dict
    observations_by_patient[patient_id][row["FIELD"]] = row["VALUE"]

patients = []

# then build the final patient dicts putting the clean data with the correct 
# each-column-is-a-field fields 
# with the bad observaitons data that has all the fields under the same
# description column.
for _, row in patients_df.iterrows():
    patient_id = row["Id"]
    observations = observations_by_patient.get(patient_id, {})
    birthdate = pd.to_datetime(row["BIRTHDATE"])
    age = dataset_end_date.year - birthdate.year
    if (dataset_end_date.month, dataset_end_date.day) < (birthdate.month, birthdate.day):
        age -= 1

    patient = {
        "id": patient_id,
        "age": age,
        "gender": row["GENDER"],
        "race": row["RACE"],
        "ethnicity": row["ETHNICITY"],
        "income": row["INCOME"],
        "education": observations.get("education"),
        "employment": observations.get("employment"),
        "housing": observations.get("housing"),
        "transportation": observations.get("transportation"),
        "social_connection": observations.get("social_connection"),
        "smoking": observations.get("smoking"),
        "bmi": observations.get("bmi"),
        "weight": observations.get("weight"),
        "systolic_bp": observations.get("systolic_bp"),
        "diastolic_bp": observations.get("diastolic_bp"),
        "conditions": conditions_by_patient.get(patient_id, []),
        "medications": medications_by_patient.get(patient_id, [])
    }

    patients.append(patient)



#################################################################################################

# Make token list for a patients, bascially this is the vocab list builder function
def patient_to_tokens(patient):
    tokens = ["<START>"]

    # AGE
    age = patient["age"]

    if age < 18:
        age_group = "0-17"
    elif age < 40:
        age_group = "18-39"
    elif age < 60:
        age_group = "40-59"
    elif age < 80:
        age_group = "60-79"
    else:
        age_group = "80+"

    tokens.append(f"AGE_{age_group}")

    # DEMOGRAPHICS
    tokens.append(f"SEX_{patient['gender']}")
    tokens.append(f"RACE_{patient['race']}")
    tokens.append(f"ETHNICITY_{patient['ethnicity']}")

    # INCOME
    income = patient["income"]

    if pd.notna(income):
        income = float(income)

        if income < 25000:
            income_group = "0-24999"
        elif income < 50000:
            income_group = "25000-49999"
        elif income < 75000:
            income_group = "50000-74999"
        elif income < 100000:
            income_group = "75000-99999"
        else:
            income_group = "100000+"

        tokens.append(f"INCOME_{income_group}")

    # SOCIAL DETERMINANTS / BEHAVIORAL
    social_fields = [
        "education",
        "employment",
        "housing",
        "transportation",
        "social_connection",
        "smoking"
    ]

    for field in social_fields:
        value = patient[field]

        if pd.notna(value):
            clean_value = str(value).strip().replace(" ", "_")
            tokens.append(f"{field.upper()}_{clean_value}")

    # BMI
    if pd.notna(patient["bmi"]):
        bmi = float(patient["bmi"])

        if bmi < 18.5:
            bmi_group = "UNDERWEIGHT"
        elif bmi < 25:
            bmi_group = "NORMAL"
        elif bmi < 30:
            bmi_group = "OVERWEIGHT"
        elif bmi < 35:
            bmi_group = "OBESE_I"
        elif bmi < 40:
            bmi_group = "OBESE_II"
        else:
            bmi_group = "OBESE_III"

        tokens.append(f"BMI_{bmi_group}")

    # BODY WEIGHT
    if pd.notna(patient["weight"]):
        weight = float(patient["weight"])

        if weight < 50:
            weight_group = "UNDER_50KG"
        elif weight < 70:
            weight_group = "50-69KG"
        elif weight < 90:
            weight_group = "70-89KG"
        elif weight < 110:
            weight_group = "90-109KG"
        else:
            weight_group = "110KG+"

        tokens.append(f"WEIGHT_{weight_group}")

    # SYSTOLIC BLOOD PRESSURE
    if pd.notna(patient["systolic_bp"]):
        systolic = float(patient["systolic_bp"])

        if systolic < 120:
            systolic_group = "UNDER_120"
        elif systolic < 130:
            systolic_group = "120-129"
        elif systolic < 140:
            systolic_group = "130-139"
        else:
            systolic_group = "140+"

        tokens.append(f"SYSTOLIC_BP_{systolic_group}")

    # DIASTOLIC BLOOD PRESSURE
    if pd.notna(patient["diastolic_bp"]):
        diastolic = float(patient["diastolic_bp"])

        if diastolic < 80:
            diastolic_group = "UNDER_80"
        elif diastolic < 90:
            diastolic_group = "80-89"
        else:
            diastolic_group = "90+"

        tokens.append(f"DIASTOLIC_BP_{diastolic_group}")

    # CONDITIONS
    for condition in patient["conditions"]:
        clean_condition = str(condition).strip().replace(" ", "_")
        tokens.append(f"CONDITION_{clean_condition}")

    # MEDICATIONS
    for medication in patient["medications"]:
        clean_medication = str(medication).strip().replace(" ", "_")
        tokens.append(f"MEDICATION_{clean_medication}")

    tokens.append("<END>")

    return tokens


# Build a vocabulary from all patient tokens
all_tokens = [patient_to_tokens(patient) for patient in patients]
# sorted(set(token for sequence in all_tokens for token in sequence))
tokens = []
for sequence in all_tokens:
    for token in sequence:
        tokens.append(token)
unique_tokens = set(tokens)       # Remove duplicates
vocab = sorted(unique_tokens) 

# Assign each unique token an integer ID
token_to_id = {token: i for i, token in enumerate(vocab)}
id_to_token = {i: token for token, i in token_to_id.items()}

# Encoding is just looking up each token's index in the vocabulary and decode reverses
def encode(tokens):
    """Convert patient tokens into integer IDs."""
    return [token_to_id[token] for token in tokens]

def decode(ids):
    """Convert integer IDs back into patient tokens."""
    return [id_to_token[i] for i in ids]


if __name__ == "__main__":
    print("Total patients:", len(patients))
    print("Vocabulary size:", len(vocab))

    tokens = patient_to_tokens(patients[0])
    ids = encode(tokens)

    print("\nFirst patient tokens:")
    print(tokens)
    print("\nSequence length:", len(tokens))
    print("Encoded length:", len(ids))
    sequence_lengths = [len(tokens) for tokens in all_tokens]

    print("\nSequence lengths")
    print("Minimum:", min(sequence_lengths))
    print("Average:", sum(sequence_lengths) / len(sequence_lengths))
    print("Maximum:", max(sequence_lengths))
