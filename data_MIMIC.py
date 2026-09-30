import pandas as pd
from pathlib import Path

DATA_PATH = Path(r"C:\Users\benpa\OneDrive\Desktop\UT MSDS\DSC AI in Healthcare\High-risk Project\project_data\MIMIC_general")

# Load basic patient and hospital admission information.
patients_df = pd.read_csv(DATA_PATH / "patients.csv.gz")
admissions_df = pd.read_csv(DATA_PATH / "admissions.csv.gz")

# Convert admission time to a date and sort admissions chronologically.
admissions_df["admittime"] = pd.to_datetime(admissions_df["admittime"])

# Inspect the clinical tables available for building patient histories.
diagnoses_df = pd.read_csv(DATA_PATH / "diagnoses_icd.csv.gz")
diagnosis_names_df = pd.read_csv(DATA_PATH / "d_icd_diagnoses.csv.gz")

procedures_df = pd.read_csv(DATA_PATH / "procedures_icd.csv.gz")
procedure_names_df = pd.read_csv(DATA_PATH / "d_icd_procedures.csv.gz")

prescriptions_df = pd.read_csv(DATA_PATH / "prescriptions.csv.gz")
omr_df = pd.read_csv(DATA_PATH / "omr.csv.gz")

# Add readable diagnosis names to each diagnosis record.
diagnoses_named = diagnoses_df.merge(
    diagnosis_names_df,
    on=["icd_code", "icd_version"],
    how="left")

# Build a list of unique diagnoses for each patient.
diagnoses_by_patient = (
    diagnoses_named
    .groupby("subject_id")["long_title"]
    .apply(lambda x: sorted(set(x.dropna())))
    .to_dict())

# Add readable procedure names to each procedure record.
procedures_named = procedures_df.merge(
    procedure_names_df,
    on=["icd_code", "icd_version"],
    how="left")

# Build a list of unique procedures for each patient.
procedures_by_patient = (
    procedures_named
    .groupby("subject_id")["long_title"]
    .apply(lambda x: sorted(set(x.dropna())))
    .to_dict())

# Build a list of unique prescribed medications for each patient.
medications_by_patient = (
    prescriptions_df
    .groupby("subject_id")["drug"]
    .apply(lambda x: sorted(set(x.dropna())))
    .to_dict())

# Keep the OMR measurements used in our patient representation.
selected_omr = omr_df[
    omr_df["result_name"].isin([
        "BMI (kg/m2)",
        "Weight (Lbs)",
        "Blood Pressure"])].copy()

# Convert chart date to a date so the latest measurement can be identified.
selected_omr["chartdate"] = pd.to_datetime(selected_omr["chartdate"])

# Keep the latest value for each measurement type for each patient.
latest_omr = (
    selected_omr
    .sort_values("chartdate")
    .drop_duplicates(
        subset=["subject_id", "result_name"],
        keep="last"))

# Build a dictionary of the latest OMR measurements for each patient.
omr_by_patient = {}

for _, row in latest_omr.iterrows():
    patient_id = row["subject_id"]
    result_name = row["result_name"]
    result_value = row["result_value"]

    if patient_id not in omr_by_patient:
        omr_by_patient[patient_id] = {}

    if result_name == "BMI (kg/m2)":
        omr_by_patient[patient_id]["bmi"] = result_value

    elif result_name == "Weight (Lbs)":
        omr_by_patient[patient_id]["weight"] = result_value

    elif result_name == "Blood Pressure":
        blood_pressure = str(result_value).split("/")

        if len(blood_pressure) == 2:
            omr_by_patient[patient_id]["systolic_bp"] = blood_pressure[0]
            omr_by_patient[patient_id]["diastolic_bp"] = blood_pressure[1]


# Keep the most recent admission for each patient.
admissions_df["admittime"] = pd.to_datetime(admissions_df["admittime"])

latest_admissions = (
    admissions_df
    .sort_values("admittime")
    .drop_duplicates(subset="subject_id", keep="last"))

# Convert the latest admission rows into a patient lookup dictionary.
admissions_by_patient = (
    latest_admissions
    .set_index("subject_id")
    [["race", "insurance", "marital_status"]]
    .to_dict("index"))

# Combine demographics, admission information, measurements,
# diagnoses, procedures and medications into one record per patient.
patients = []

for _, row in patients_df.iterrows():
    patient_id = row["subject_id"]

    admission = admissions_by_patient.get(patient_id, {})
    omr = omr_by_patient.get(patient_id, {})

    patient = {
        "id": patient_id,
        "age": row["anchor_age"],
        "gender": row["gender"],
        "race": admission.get("race"),
        "insurance": admission.get("insurance"),
        "marital_status": admission.get("marital_status"),
        "bmi": omr.get("bmi"),
        "weight": omr.get("weight"),
        "systolic_bp": omr.get("systolic_bp"),
        "diastolic_bp": omr.get("diastolic_bp"),
        "diagnoses": diagnoses_by_patient.get(patient_id, []),
        "procedures": procedures_by_patient.get(patient_id, []),
        "medications": medications_by_patient.get(patient_id, [])}

    patients.append(patient)

# Keep patients whose clinical history fits comfortably within
# the transformer's 128-token context window.
MAX_CLINICAL_HISTORY = 100

filtered_patients = []

for patient in patients:
    clinical_history_length = (
        len(patient["diagnoses"])
        + len(patient["procedures"])
        + len(patient["medications"]) )

    if clinical_history_length <= MAX_CLINICAL_HISTORY:
        filtered_patients.append(patient)

patients = filtered_patients

print("\nPatients after history-length filter:", len(patients))

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

    if pd.notna(patient["race"]):
        tokens.append(f"RACE_{str(patient['race']).strip().replace(' ', '_')}")
    if pd.notna(patient["insurance"]):
        tokens.append(f"INSURANCE_{str(patient['insurance']).strip().replace(' ', '_')}")
    if pd.notna(patient["marital_status"]):
        tokens.append(f"MARITAL_STATUS_{str(patient['marital_status']).strip().replace(' ', '_')}")

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

        if weight < 110:
            weight_group = "UNDER_110LBS"
        elif weight < 155:
            weight_group = "110-154LBS"
        elif weight < 200:
            weight_group = "155-199LBS"
        elif weight < 245:
            weight_group = "200-244LBS"
        else:
            weight_group = "245LBS+"

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
    # DIAGNOSE
    for diagnosis in patient["diagnoses"]:
        clean_diagnosis = str(diagnosis).strip().replace(" ", "_")
        tokens.append(f"DIAGNOSIS_{clean_diagnosis}")
    # PROCEDURES
    for procedure in patient["procedures"]:
        clean_procedure = str(procedure).strip().replace(" ", "_")
        tokens.append(f"PROCEDURE_{clean_procedure}")
    # MEDICAITONS
    for medication in patient["medications"]:
        clean_medication = str(medication).strip().replace(" ", "_")
        tokens.append(f"MEDICATION_{clean_medication}")
    tokens.append("<END>")
    return tokens


all_tokens = [patient_to_tokens(patient) for patient in patients]

vocab = sorted(set(
    token
    for sequence in all_tokens
    for token in sequence))

token_to_id = {token:i for i, token in enumerate(vocab)}
id_to_token = {i:token for token, i in token_to_id.items()}

def encode(tokens):
    return [token_to_id[token] for token in tokens]

def decode(ids):
    return [id_to_token[i] for i in ids]

sequence_lengths = [len(sequence) for sequence in all_tokens]

print("\nVocabulary size:", len(vocab))
print("Minimum sequence length:", min(sequence_lengths))
print("Average sequence length:", sum(sequence_lengths) / len(sequence_lengths))
print("Maximum sequence length:", max(sequence_lengths))

print("\nFirst patient tokens:")
print(all_tokens[0])

