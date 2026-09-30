import pandas as pd
import random
from pathlib import Path
from data_synthea import patients, patient_to_tokens
import matplotlib.pyplot as plt
import numpy as np



SEED = 42
DATA_PATH = Path( r"C:\Users\benpa\OneDrive\Desktop\UT MSDS\DSC AI in Healthcare\High-risk Project\project_data\Synthea")

# repeat sample method with the same random seed to gaurantee the same patients are selected for comparison
########################################################################
# PRIVACY ASSESSMENT
# This uses exact duplicaiotn and nearest similarity indicators
random.seed(SEED)
shuffled_patients = patients.copy()
random.shuffle(shuffled_patients)

datasets = {
    "100": {
        "training": shuffled_patients[:100],
        "generated": DATA_PATH / "generated_100" / "generated_patients.csv"},
    "1k": {
        "training": shuffled_patients[:1000],
        "generated": DATA_PATH / "generated_1k" / "generated_patients.csv"},
    "10k": {
        "training": shuffled_patients[:10000],
        "generated": DATA_PATH / "generated_10k" / "generated_patients.csv"}}

# format to compare token lists for training and generated
for name, data in datasets.items():
    data["training_tokens"] = [ patient_to_tokens(patient) for patient in data["training"]]
    generated_df = pd.read_csv(data["generated"])
    data["generated_tokens"] = [ tokens.split("|") for tokens in generated_df["tokens"] ]
    print( name, "training:", len(data["training_tokens"]), "generated:", len(data["generated_tokens"]))

print("\nPRIVACY / MEMORIZATION RISK")
print("\nExact training-record matches")

for name, data in datasets.items():
    # compare and use set vs list
    training_records = {tuple(tokens) for tokens in data["training_tokens"]}
    generated_records = [tuple(tokens) for tokens in data["generated_tokens"]]

    # percent exact match
    exact_matches = sum(  record in training_records for record in generated_records)
    exact_match_rate = exact_matches / len(generated_records) * 100

    data["exact_matches"] = exact_matches
    data["exact_match_rate"] = exact_match_rate

    print( f"{name}: {exact_matches}/{len(generated_records)} "f"({exact_match_rate:.2f}%)" )
"""
PRIVACY / MEMORIZATION RISK
Exact training-record matches
100: 215/5000 (4.30%)
1k: 39/5000 (0.78%)
10k: 113/5000 (2.26%)

All training set sizes resulted in exact matches and memorization was encountered
in all generative models.
Exact memorization was highest with 100 training patients (4.30%) and
dropped substantially with 1,000 (0.78%). The increase at 10,000 (2.26%)
shows that training size alone does not determine exact-match risk.
"""

print("\nNearest training-record similarity")

for name, data in datasets.items():
    training_sets = [set(tokens) for tokens in data["training_tokens"]]
    similarities = []

    for generated_tokens in data["generated_tokens"]:
        generated_set = set(generated_tokens)
        best_similarity = 0.0

        for training_set in training_sets:
            intersection = len(generated_set & training_set)
            union = len(generated_set | training_set)
            similarity = intersection / union
            if similarity > best_similarity:
                best_similarity = similarity

        similarities.append(best_similarity)

    # Jaccard:
    # take the highest token match set for each generated patient compared to each training patient
    # generated_patient{X,Y,M}, training_patient_1 {X,Y,Z} thats 2/3, for training data 1 do this 
    # comaprison for every single training person and take the highest overlap
    # take teh average of these similarities and the rate that had high similarity
    average_similarity = sum(similarities) / len(similarities)
    high_similarity_rate = sum(x >= 0.90 for x in similarities) / len(similarities) * 100

    data["average_similarity"] = average_similarity
    data["high_similarity_rate"] = high_similarity_rate

    print( f"{name}: average nearest similarity = {average_similarity:.3f}, "
        f">=90% similar = {high_similarity_rate:.2f}%")

"""
Nearest training-record similarity
100: average nearest similarity = 0.530, >=90% similar = 5.40%
1k: average nearest similarity = 0.537, >=90% similar = 1.24%
10k: average nearest similarity = 0.609, >=90% similar = 3.22%

On average, each generated patient had a closest training patient with 
roughly 53-61% token-set similarity.

Memorization risk was highest for the 100-patient model and lowest for
the 1,000-patient model. Risk increased again at 10,000, showing that
larger training populations did not monotonically reduce similarity.

the 10k model has 10x more possible training records to be similar to than 
the 1k model, so nearest-neighbour similarity naturally has more opportunities 
to find a close match. That means we should interpret the trend carefully 
rather than saying the 10k model necessarily "memorized more." 
Exact-match rate is less affected by that issue, though common Synthea 
profiles could still recur naturally.
"""
labels = ["100", "1,000", "10,000"]
exact_rates = [datasets["100"]["exact_match_rate"],
               datasets["1k"]["exact_match_rate"],
               datasets["10k"]["exact_match_rate"]]
high_similarity_rates = [datasets["100"]["high_similarity_rate"],
                         datasets["1k"]["high_similarity_rate"],
                         datasets["10k"]["high_similarity_rate"]]
x = range(len(labels))
width = 0.35
plt.figure(figsize=(8, 5))
plt.bar([i - width/2 for i in x], exact_rates, width, label="Exact match")
plt.bar([i + width/2 for i in x], high_similarity_rates, width, label="≥90% similar")
plt.xticks(x, labels)
plt.xlabel("Training population size")
plt.ylabel("Generated records (%)")
plt.title("Synthea Privacy / Memorization Risk")
plt.legend()
plt.tight_layout()


########################################################################
# POPULAITON REPRESENTATION

print("\nPOPULATION REPRESENTATION")
print("\nMarginal prevalence error")

def token_prevalence(records):
    counts = {}
    for record in records:
        for token in set(record):
            if token not in {"<START>", "<END>"}:
                counts[token] = counts.get(token, 0) + 1
    return {token: count / len(records) for token, count in counts.items()}
# Compare prevalence of every token in training and generated populations.
for name, data in datasets.items():
    training_prevalence = token_prevalence(data["training_tokens"])
    generated_prevalence = token_prevalence(data["generated_tokens"])
    all_tokens = set(training_prevalence) | set(generated_prevalence)
    errors = [ abs(training_prevalence.get(token, 0) - generated_prevalence.get(token, 0)) for token in all_tokens]
    marginal_error = sum(errors) / len(errors) * 100
    data["marginal_error"] = marginal_error
    data["training_prevalence"] = training_prevalence
    print(f"\n{name}: mean absolute prevalence error = {marginal_error:.2f} percentage points")
    print("Top 10 training variables:")

    top_10 = sorted(training_prevalence.items(), key=lambda x: x[1], reverse=True)[:10]

    for token, train_prev in top_10:
        gen_prev = generated_prevalence.get(token, 0)
        print(  f"{token:35} " f"training={train_prev*100:6.2f}%  " f"generated={gen_prev*100:6.2f}%" )

# CORRELATION STRUCTURE

print("\nCORRELATION STRUCTURE")

def binary_matrix(records, variables):
    rows = []
    for record in records:
        record_set = set(record)
        rows.append({
            variable: int(variable in record_set)
            for variable in variables })
    return pd.DataFrame(rows)

for name, data in datasets.items():
    prevalence = data["training_prevalence"]
    # Select the 10 top most prevalent characteristics that vary across patients
    candidates = [(token, prev) for token, prev in prevalence.items()  if 0 < prev < 1]

    top_10 = [ token
        for token, prev in sorted(
            candidates,
            key=lambda x: x[1],
            reverse=True  )[:10]  ]

    print(f"\n{name}: selected variables")
    for token in top_10:
        print(f"  {token}: {prevalence[token] * 100:.2f}%")

    # Represent each characteristic as present=1 or absent=0.
    train_matrix = binary_matrix(data["training_tokens"], top_10)
    gen_matrix = binary_matrix(data["generated_tokens"], top_10)

    # Correlation requires variation in both populations.
    valid = [
        variable for variable in top_10
        if train_matrix[variable].nunique() > 1
        and gen_matrix[variable].nunique() > 1 ]

    train_matrix = train_matrix[valid]
    gen_matrix = gen_matrix[valid]
    train_corr = train_matrix.corr(method="spearman")
    gen_corr = gen_matrix.corr(method="spearman")

    # Compare corresponding off-diagonal correlations.
    difference = train_corr.values - gen_corr.values
    mask = ~np.eye(len(valid), dtype=bool)
    correlation_error = np.sqrt(np.nanmean(difference[mask] ** 2))
    data["train_corr"] = train_corr
    data["gen_corr"] = gen_corr
    data["correlation_error"] = correlation_error

    print(  f"{name}: variables = {len(valid)}, "  f"correlation matrix RMSE = {correlation_error:.3f}"  )

    # Plot training and generated correlation matrices.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im = axes[0].imshow(
        train_corr,
        vmin=-1,
        vmax=1,
        cmap="coolwarm"  )
    axes[0].set_title(f"Training Population ({name})")
    axes[1].imshow(
        gen_corr,
        vmin=-1,
        vmax=1,
        cmap="coolwarm" )
    
    axes[1].set_title(f"Generated Population ({name})")

    for ax in axes:
        ax.set_xticks(range(len(valid)))
        ax.set_yticks(range(len(valid)))
        ax.set_xticklabels(valid, rotation=90, fontsize=7)
        ax.set_yticklabels(valid, fontsize=7)

    fig.colorbar( im, ax=axes, label="Spearman correlation", shrink=0.8 )
    fig.suptitle(
        f"Synthea Correlation Structure — {name} Training Patients\n"
        f"Correlation Matrix RMSE = {correlation_error:.3f}" )
    plt.subplots_adjust( bottom=0.35,  top=0.82, wspace=0.30  )
    plt.savefig(  f"synthea_correlation_{name}.png",  dpi=300, bbox_inches="tight" )
    plt.show()

########################################################################
# STRUCTURAL FIDELITY / QUALITY

print("\nSTRUCTURAL FIDELITY / QUALITY")

required_fields = [  "AGE_",  "SEX_", "RACE_", "ETHNICITY_"]

single_value_fields = [
    "AGE_",
    "SEX_",
    "RACE_",
    "ETHNICITY_",
    "INCOME_",
    "EDUCATION_",
    "EMPLOYMENT_",
    "HOUSING_",
    "TRANSPORTATION_",
    "SOCIAL_CONNECTION_",
    "SMOKING_",
    "BMI_",
    "WEIGHT_",
    "SYSTOLIC_BP_",
    "DIASTOLIC_BP_"]

def count_prefix(tokens, prefix):
    return sum(token.startswith(prefix) for token in tokens)

def check_record(tokens):
    errors = {
        "bad_start_end": False,
        "required_field_error": False,
        "multiple_single_value": False,
        "duplicate_clinical_token": False}

    if not tokens or tokens[0] != "<START>" or tokens[-1] != "<END>":
        errors["bad_start_end"] = True

    for prefix in required_fields:
        if count_prefix(tokens, prefix) != 1:
            errors["required_field_error"] = True
            break

    for prefix in single_value_fields:
        if count_prefix(tokens, prefix) > 1:
            errors["multiple_single_value"] = True
            break

    clinical_tokens = [
        token for token in tokens
        if token.startswith("CONDITION_")
        or token.startswith("MEDICATION_") ]

    if len(clinical_tokens) != len(set(clinical_tokens)):
        errors["duplicate_clinical_token"] = True

    return any(errors.values()), errors

for name, data in datasets.items():
    records = data["generated_tokens"]
    invalid_count = 0
    error_counts = {
        "bad_start_end": 0,
        "required_field_error": 0,
        "multiple_single_value": 0,
        "duplicate_clinical_token": 0 }

    for tokens in records:
        invalid, errors = check_record(tokens)
        if invalid:
            invalid_count += 1
        for error, occurred in errors.items():
            if occurred:
                error_counts[error] += 1

    total = len(records)
    unique_count = len({tuple(record) for record in records})
    invalid_rate = invalid_count / total * 100
    unique_rate = unique_count / total * 100
    data["invalid_rate"] = invalid_rate
    data["unique_rate"] = unique_rate

    print(f"\n{name}:")
    print(f"  Total generated records: {total}")
    print(f"  Invalid records: {invalid_count} ({invalid_rate:.2f}%)")
    print(f"  Unique records: {unique_count} ({unique_rate:.2f}%)")
    print("  Invalidity breakdown:")

    for error, count in error_counts.items():
        print(f"    {error}: {count} ({count / total * 100:.2f}%)")

