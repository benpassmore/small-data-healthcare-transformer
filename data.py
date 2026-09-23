

patients = [
    {
        "age_group": "40-59",
        "sex": "F",
        "conditions": ["hypertension", "diabetes"],
    },
    {
        "age_group": "20-39",
        "sex": "M",
        "conditions": ["asthma"],
    },
    {
        "age_group": "60-79",
        "sex": "F",
        "conditions": ["hypertension"],
    },
]

def patient_to_tokens(patient):
    """Convert one patient into a sequence of categorical tokens."""
    tokens = [ "<START>", f"AGE_{patient['age_group']}", f"SEX_{patient['sex']}",]
    for condition in sorted(patient["conditions"]):
        tokens.append(f"CONDITION_{condition}")
    tokens.append("<END>")
    return tokens

if __name__ == "__main__":
    for patient in patients:
        print(patient_to_tokens(patient))
