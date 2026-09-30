import torch
import pandas as pd
from pathlib import Path
from model_synthea import PatientTransformer, CONTEXT_LENGTH

# Generation configuration
NUM_PATIENTS = 5000
TEMPERATURE = 1.0

# generated Synthea patients 
DATA_PATH = Path( r"C:\Users\benpa\OneDrive\Desktop\UT MSDS\DSC AI in Healthcare\High-risk Project\project_data\Synthea")

# Each trained model and its corresponding output folder.
models = [
    ("synthea_100.pt", "generated_100"),
    ("synthea_1k.pt", "generated_1k"),
    ("synthea_10k.pt", "generated_10k")]

# Generate one patient at a time.
def generate_patient(model, token_to_id, id_to_token):
    start_id = token_to_id["<START>"]
    end_id = token_to_id["<END>"]
    # Begin with the START token.
    generated_ids = [start_id]

    # No gradients are needed during generation.
    with torch.no_grad():
        for step in range(CONTEXT_LENGTH - 1):

            # Convert the generated sequence into a model input.
            inputs = torch.tensor([generated_ids], dtype=torch.long)
            # Forward pass - predict the next token at every position.
            logits = model(inputs)
            # Only the final position predicts our next token.
            next_token_logits = logits[0, -1, :]
            # Adjust randomness and convert scores into probabilities.
            next_token_logits = next_token_logits / TEMPERATURE
            probabilities = torch.softmax(next_token_logits, dim=-1)
            # Sample one token from the probability distribution.
            next_token_id = torch.multinomial(probabilities,num_samples=1).item()
            # Append the sampled token to our patient sequence.
            generated_ids.append(next_token_id)
            # Stop once the model generates END.
            if next_token_id == end_id:
                break
    # Convert numerical IDs back into readable patient tokens.
    return [id_to_token[token_id] for token_id in generated_ids]


# Generate patients from each of our three trained models.
for checkpoint_name, output_folder in models:

    print(f"\nLoading {checkpoint_name}")
    # Load our trained model and vocabulary.
    checkpoint = torch.load(
        checkpoint_name,
        map_location="cpu",
        weights_only=True)

    model = PatientTransformer()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    token_to_id = checkpoint["token_to_id"]
    id_to_token = {
        i: token for token, i in token_to_id.items() }
    generated_patients = []
    # Generate the requested number of patients.
    for patient_num in range(NUM_PATIENTS):

        patient = generate_patient(model, token_to_id, id_to_token)
        # Store the complete generated sequence without changing it.
        generated_patients.append({
            "patient_number": patient_num + 1,
            "tokens": "|".join(patient),
            "sequence_length": len(patient)})
        # Display progress every 500 generated patients.
        if (patient_num + 1) % 500 == 0:
            print( f"{checkpoint_name}: " f"{patient_num + 1}/{NUM_PATIENTS}" )

    # Create the output folder 
    output_path = DATA_PATH / output_folder
    output_path.mkdir(parents=True, exist_ok=True)
    # Save generated patients as a CSV file.
    output_file = output_path / "generated_patients.csv"
    generated_df = pd.DataFrame(generated_patients)
    generated_df.to_csv(output_file, index=False)
    print(f"Saved {output_file}")

print("\nGeneration complete.")