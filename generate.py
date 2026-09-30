
import torch
from data import id_to_token
from model import PatientTransformer, CONTEXT_LENGTH

# Generation configuration
NUM_PATIENTS = 5
TEMPERATURE = 1.0

# Load our trained model and vocabulary.
checkpoint = torch.load("checkpoint.pt", map_location="cpu", weights_only=True)

model = PatientTransformer()
model.load_state_dict(checkpoint["model_state"])
model.eval()

token_to_id = checkpoint["token_to_id"]
id_to_token = {i: token for token, i in token_to_id.items()}

start_id = token_to_id["<START>"]
end_id = token_to_id["<END>"]

# Generate one patient at a time.
def generate_patient():
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
            next_token_id = torch.multinomial(probabilities, num_samples=1).item()
            # Append the sampled token to our patient sequence.
            generated_ids.append(next_token_id)
            # Stop once the model generates END.
            if next_token_id == end_id:
                break

    # Convert numerical IDs back into readable patient tokens.
    return [id_to_token[token_id] for token_id in generated_ids]

# Generate and display five patients.
if __name__ == "__main__":
    for patient_num in range(NUM_PATIENTS):
        patient = generate_patient()
        print(f"Patient {patient_num + 1}: {patient}")
