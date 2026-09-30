
import torch
import torch.nn.functional as F
from data import patients, patient_to_tokens, encode, token_to_id
from model import PatientTransformer

"""
Prepare all three patients as input/target sequences.
Initialize our PatientTransformer and AdamW optimizer.
Run the forward pass and calculate cross-entropy loss.
Backpropagate and update the model's weights.
Save the trained model so generate.py can use it.
"""

# Training configuration
EPOCHS = 500
LEARNING_RATE = 0.001

# Initialize our randomly weighted transformer and optimizer
torch.manual_seed(42)
model = PatientTransformer()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Prepare the training data.
# Each patient provides input tokens and their next-token targets
training_data = []
for patient in patients:
    tokens = patient_to_tokens(patient)
    ids = encode(tokens)
    # Shift the targets one position ahead of the inputs
    input_ids = ids[:-1]
    target_ids = ids[1:]
    # Convert both sequences into PyTorch tensors
    inputs = torch.tensor([input_ids], dtype=torch.long)
    targets = torch.tensor([target_ids], dtype=torch.long)
    training_data.append((inputs, targets))

# Train the transformer
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    for inputs, targets in training_data:

        # Forward pass - produce next-token logits.
        # Shape - [1, sequence_length, vocabulary_size]
        logits = model(inputs)

        # Calculate cross-entropy loss against the actual next tokens
        # Flatten the batch and sequence dimensions for cross_entropy
        predictions = logits.reshape(-1, logits.size(-1))
        actual_tokens = targets.reshape(-1)
        loss = F.cross_entropy(predictions, actual_tokens)
        # Backpropagation-calculate gradients for all model parameters
        optimizer.zero_grad()
        loss.backward()
        # Optimization to update the model's weights
        optimizer.step()
        total_loss += loss.item()

    # Display average patient loss periodically
    if (epoch + 1) % 50 == 0:
        average_loss = total_loss / len(training_data)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {average_loss:.4f}")

# Save the trained model and its vocabulary
checkpoint = { "model_state": model.state_dict(), "token_to_id": token_to_id}

torch.save(checkpoint, "checkpoint.pt")
print("\nTraining complete. Model saved to checkpoint.pt")