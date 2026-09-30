

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

# Make token list for a patients, bascially this is the vocab list builder function
def patient_to_tokens(patient):
    """Convert one patient into a sequence of categorical tokens."""
    tokens = [ "<START>", f"AGE_{patient['age_group']}", f"SEX_{patient['sex']}",]
    for condition in sorted(patient["conditions"]):
        tokens.append(f"CONDITION_{condition}")
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
    print("Vocabulary:", token_to_id)
    print("Vocabulary size:", len(vocab))

    for patient in patients:
        tokens = patient_to_tokens(patient)
        ids = encode(tokens)
        print("\nTokens:", tokens)
        print("IDs:", ids)
        print("Decoded:", decode(ids))

