import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path
from data_MIMIC import patients, patient_to_tokens

SEED = 42

DATA_PATH = Path( r"C:\Users\benpa\OneDrive\Desktop\UT MSDS\DSC AI in Healthcare\High-risk Project\project_data\MIMIC_general")

# Reconstruct the exact nested training populations.
random.seed(SEED)
shuffled_patients = patients.copy()
random.shuffle(shuffled_patients)

datasets = {
    "100": { "training": shuffled_patients[:100],
        "generated": DATA_PATH / "generated_100" / "generated_patients.csv" },
    "1k": {
        "training": shuffled_patients[:1000],
        "generated": DATA_PATH / "generated_1k" / "generated_patients.csv" },
    "10k": {
        "training": shuffled_patients[:10000],
        "generated": DATA_PATH / "generated_10k" / "generated_patients.csv" }}

# Convert training and generated patients to comparable token lists
for name, data in datasets.items():
    data["training_tokens"] = [
        patient_to_tokens(patient) for patient in data["training"]]

    generated_df = pd.read_csv(data["generated"])
    data["generated_tokens"] = [ tokens.split("|") for tokens in generated_df["tokens"] ]

    print( name, "training:", len(data["training_tokens"]), "generated:", len(data["generated_tokens"]))

########################################################################
# PRIVACY /  RISK

print("\nPRIVACY / MEMORIZATION RISK")
print("\nExact training-record matches")

for name, data in datasets.items():
    training_records = {tuple(tokens) for tokens in data["training_tokens"]}
    generated_records = [
        tuple(tokens) for tokens in data["generated_tokens"]]
    exact_matches = sum( record in training_records for record in generated_records )
    exact_match_rate = ( exact_matches / len(generated_records) * 100 )
    data["exact_matches"] = exact_matches
    data["exact_match_rate"] = exact_match_rate
    print(f"{name}: {exact_matches}/{len(generated_records)} "
        f"({exact_match_rate:.2f}%)" )

print("\nNearest-training-record Jaccard similarity")

def nearest_jaccard(generated_records, training_records):
    training_sets = [
        set(record) for record in training_records]

    scores = []
    for i, generated in enumerate(generated_records):
        generated_set = set(generated)
        best_similarity = 0.0
        for training_set in training_sets:
            intersection = len(generated_set & training_set)
            # needed for Jaccard
            union = len(generated_set | training_set )

            if union > 0:
                similarity = intersection / union
                if similarity > best_similarity:
                    best_similarity = similarity

        scores.append(best_similarity)

        if (i + 1) % 500 == 0:
            print(f"  compared {i + 1}/" f"{len(generated_records)} generated patients")
    return scores

for name, data in datasets.items():
    print(f"\n{name}")

    similarities = nearest_jaccard( data["generated_tokens"], data["training_tokens"] )
    average_similarity = np.mean(similarities)
    high_similarity_rate = (
        sum(score >= 0.90 for score in similarities)
        / len(similarities) * 100 )

    data["average_similarity"] = average_similarity
    data["high_similarity_rate"] = high_similarity_rate

    print( f"Average nearest similarity: " f"{average_similarity:.3f}" )
    print( f"≥90% similar: "
        f"{high_similarity_rate:.2f}%" )

# Privacy visualization.
labels = ["100", "1,000", "10,000"]

exact_rates = [
    datasets["100"]["exact_match_rate"],
    datasets["1k"]["exact_match_rate"],
    datasets["10k"]["exact_match_rate"]]

high_similarity_rates = [
    datasets["100"]["high_similarity_rate"],
    datasets["1k"]["high_similarity_rate"],
    datasets["10k"]["high_similarity_rate"]]

x = np.arange(len(labels))
width = 0.35
plt.figure(figsize=(8, 5))
plt.bar(x - width/2,exact_rates,width,label="Exact match")
plt.bar( x + width/2, high_similarity_rates,
    width, label="≥90% similar")
plt.xticks(x, labels)
plt.xlabel("Training population size")
plt.ylabel("Generated records (%)")
plt.title("MIMIC Privacy / Memorization Risk")
plt.legend()
plt.tight_layout()
plt.savefig( "mimic_privacy.png", dpi=300, bbox_inches="tight")
plt.show()

########################################################################
# POP Representation


print("\nPOPULATION REPRESENTATION")
print("\nMarginal prevalence error")

def token_prevalence(records):
    counts = {}

    for record in records:
        for token in set(record):
            if token not in {"<START>", "<END>"}:
                counts[token] = counts.get(token, 0) + 1
    return {
        token: count / len(records)
        for token, count in counts.items() }

# Compare prevalence of every token in training and generated populations
for name, data in datasets.items():
    training_prevalence = token_prevalence(
        data["training_tokens"] )

    generated_prevalence = token_prevalence( data["generated_tokens"] )
    all_tokens = (
        set(training_prevalence)
        | set(generated_prevalence) )

    errors = [
        abs(training_prevalence.get(token, 0)- generated_prevalence.get(token, 0) )
        for token in all_tokens]
    marginal_error = ( sum(errors) / len(errors) * 100)
    data["marginal_error"] = marginal_error
    data["training_prevalence"] = training_prevalence
    print( f"\n{name}: mean absolute prevalence error = " f"{marginal_error:.2f} percentage points")
    print("Top 10 training variables:")

    top_10 = sorted(
        training_prevalence.items(),
        key=lambda x: x[1],
        reverse=True )[:10]

    for token, train_prev in top_10:
        gen_prev = generated_prevalence.get( token, 0 )

        print(  f"{token:45} " f"training={train_prev*100:6.2f}%  "  f"generated={gen_prev*100:6.2f}%" )

########################################################################
# Corr Structure and Frobenius 

print("\nCORRELATION STRUCTURE")

def binary_matrix(records, variables):
    rows = []
    for record in records:
        record_set = set(record)

        rows.append({variable: int(variable in record_set)
            for variable in variables })

    return pd.DataFrame(rows)

for name, data in datasets.items():
    prevalence = data["training_prevalence"]

    # Select the 10 most prevalent characteristics that vary.
    candidates = [(token, prev)
        for token, prev in prevalence.items()
        if 0 < prev < 1 ]

    top_10 = [
        token
        for token, prev in sorted(
            candidates,
            key=lambda x: x[1],
            reverse=True )[:10]]

    print(f"{name}: selected variables")
    for token in top_10:
        print( f"  {token}: "  f"{prevalence[token] * 100:.2f}%" )

    # Represent each characteristic as simple existence present=1 or absent=0
    train_matrix = binary_matrix( data["training_tokens"], top_10 )
    gen_matrix = binary_matrix( data["generated_tokens"],top_10 )

    # Correlation requires variation in both populations
    valid = [
        variable
        for variable in top_10
        if train_matrix[variable].nunique() > 1
        and gen_matrix[variable].nunique() > 1]
    train_matrix = train_matrix[valid]
    gen_matrix = gen_matrix[valid]
    # do the corr
    train_corr = train_matrix.corr(  method="spearman" )
    gen_corr = gen_matrix.corr( method="spearman" )
    # Compare corresponding off-diagonal correlations.
    difference = ( train_corr.values- gen_corr.values )
    # mask diag
    mask = ~np.eye( len(valid), dtype=bool)
    correlation_error = np.sqrt( np.nanmean( difference[mask] ** 2 ) )
    data["train_corr"] = train_corr
    data["gen_corr"] = gen_corr
    data["correlation_error"] = correlation_error
    print( f"{name}: variables = {len(valid)}, "
        f"correlation matrix RMSE = "
        f"{correlation_error:.3f}" )

    # plot training and generated correlation matrices
    fig, axes = plt.subplots(  1,2, figsize=(14, 6))
    im = axes[0].imshow( train_corr, vmin=-1, vmax=1, cmap="coolwarm" )
    axes[0].set_title( f"Training Population ({name})" )
    axes[1].imshow(  gen_corr,vmin=-1,   vmax=1,   cmap="coolwarm")
    axes[1].set_title(  f"Generated Population ({name})")

    for ax in axes:
        ax.set_xticks(range(len(valid)) )
        ax.set_yticks( range(len(valid)) )
        ax.set_xticklabels( valid, rotation=90,  fontsize=7 )
        ax.set_yticklabels(valid,fontsize=7 )
    fig.colorbar( im, ax=axes,  label="Spearman correlation", shrink=0.8 )
    fig.suptitle(
        f"MIMIC Correlation Structure — "
        f"{name} Training Patients\n"
        f"Correlation Matrix RMSE = "
        f"{correlation_error:.3f}" )
    plt.subplots_adjust( bottom=0.35, top=0.82, wspace=0.30 )
    plt.savefig(  f"mimic_correlation_{name}.png",  dpi=300,  bbox_inches="tight" )
    plt.show()

########################################################################
# Fidelity

print("\nSTRUCTURAL FIDELITY / QUALITY")

# Age and sex are always present in the real MIMIC representation
required_fields = [ "AGE_", "SEX_"]
# These fields can have at most one value per patient
single_value_fields = [
    "AGE_",
    "SEX_",
    "RACE_",
    "INSURANCE_",
    "MARITAL_STATUS_",
    "BMI_",
    "WEIGHT_",
    "SYSTOLIC_BP_",
    "DIASTOLIC_BP_"]

def count_prefix(tokens, prefix):
    return sum( token.startswith(prefix) for token in tokens )

def check_record(tokens):
    errors = {
        "bad_start_end": False,
        "required_field_error": False,
        "multiple_single_value": False,
        "duplicate_clinical_token": False}

    if (
        not tokens
        or tokens[0] != "<START>"
        or tokens[-1] != "<END>"):
        errors["bad_start_end"] = True
    # Required fields must occur exactly one time
    for prefix in required_fields:
        if count_prefix(tokens, prefix) != 1:
            errors["required_field_error"] = True
            break
    # Single-valued fields cannot occur more than once
    for prefix in single_value_fields:
        if count_prefix(tokens, prefix) > 1:
            errors["multiple_single_value"] = True
            break
    # Diagnoses, procedures and medications are deduplicated
    # in the original patient representation
    clinical_tokens = [
        token
        for token in tokens
        if token.startswith("DIAGNOSIS_")
        or token.startswith("PROCEDURE_")
        or token.startswith("MEDICATION_") ]

    if ( len(clinical_tokens)
        != len(set(clinical_tokens)) ):
        errors["duplicate_clinical_token"] = True
    invalid = any( errors.values() )
    return invalid, errors

for name, data in datasets.items():
    records = data["generated_tokens"]
    invalid_count = 0
    error_counts = {
        "bad_start_end": 0,
        "required_field_error": 0,
        "multiple_single_value": 0,
        "duplicate_clinical_token": 0 }

    for tokens in records:
        invalid, errors = check_record( tokens )
        if invalid:
            invalid_count += 1
        for error, occurred in errors.items():
            if occurred:
                error_counts[error] += 1
    total = len(records)
    unique_count = len({ tuple(record)for record in records })
    invalid_rate = (invalid_count / total * 100)
    unique_rate = (unique_count / total * 100)
    data["invalid_rate"] = invalid_rate
    data["unique_rate"] = unique_rate

    print(f"\n{name}:")
    print(f"  Total generated records: "
        f"{total}" )
    print( f"  Invalid records: " f"{invalid_count} "
        f"({invalid_rate:.2f}%)" )
    print( f"  Unique records: " f"{unique_count} "
        f"({unique_rate:.2f}%)" )
    print("  Invalidity breakdown:")
    for error, count in error_counts.items():
        print( f"    {error}: "
            f"{count} "
            f"({count / total * 100:.2f}%)"  )


# plot

invalid_rates = [ datasets["100"]["invalid_rate"],
    datasets["1k"]["invalid_rate"],
    datasets["10k"]["invalid_rate"]]

unique_rates = [datasets["100"]["unique_rate"],
    datasets["1k"]["unique_rate"],
    datasets["10k"]["unique_rate"]]

x = np.arange(len(labels))
plt.figure(figsize=(8, 5))
plt.bar( x - width/2,invalid_rates, width,label="Invalid records")
plt.bar( x + width/2,  unique_rates,  width, label="Unique records")
plt.xticks(x, labels)
plt.xlabel("Training population size")
plt.ylabel("Generated records (%)")
plt.title("MIMIC Structural Fidelity / Quality")
plt.legend()
plt.tight_layout()
plt.savefig("mimic_fidelity.png", dpi=300, bbox_inches="tight")
plt.show()
