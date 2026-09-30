import torch
import torch.nn.functional as F
import random
from data_MIMIC import patients, patient_to_tokens, encode, token_to_id
from model_MIMIC import PatientTransformer

EPOCHS = 50
LEARNING_RATE = 0.001
SEED = 42

# Shuffle once so the datasets are random but nested:
# first 100 are inside first 1,000, which are inside first 10,000
random.seed(SEED)
shuffled_patients = patients.copy()
random.shuffle(shuffled_patients)

dataset_sizes = [(100, "mimic_100.pt"),
    (1000, "mimic_1k.pt"),
    (10000, "mimic_10k.pt")]

for dataset_size, checkpoint_name in dataset_sizes:
    print(f"\nTraining model on {dataset_size} patients")
    training_patients = shuffled_patients[:dataset_size]
    training_data = []
    # Prepare the training data.
    # Each patient provides input tokens and their next-token targets.
    for patient in training_patients:
        tokens = patient_to_tokens(patient)
        ids = encode(tokens)
        # Shift the targets one position ahead of the inputs.
        input_ids = ids[:-1]
        target_ids = ids[1:]
        # Convert both sequences into PyTorch tensors.
        inputs = torch.tensor([input_ids], dtype=torch.long)
        targets = torch.tensor([target_ids], dtype=torch.long)
        training_data.append((inputs, targets))

    # Fresh model for each dataset size
    # Initialize our randomly weighted transformer and optimizer.
    torch.manual_seed(SEED)
    model = PatientTransformer()
    optimizer = torch.optim.AdamW( model.parameters(),lr=LEARNING_RATE)
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for inputs, targets in training_data:
            # Forward pass - produce next-token logits.
            # Shape - [1, sequence_length, vocabulary_size].
            logits = model(inputs)
            predictions = logits.reshape(-1, logits.size(-1))
            actual_tokens = targets.reshape(-1)
            # Calculate cross-entropy loss against the actual next tokens.
            # Flatten the batch and sequence dimensions for cross_entropy.
            loss = F.cross_entropy(predictions, actual_tokens)
            # Backpropagation-calculate gradients for all model parameters.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Optimization to update the model's weights.
            total_loss += loss.item()
        # Display average patient loss periodically.
        if (epoch + 1) % 50 == 0:
            average_loss = total_loss / len(training_data)

            print(
                f"Patients: {dataset_size} | "
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Loss: {average_loss:.4f}")

    checkpoint = {
        "model_state": model.state_dict(),
        "token_to_id": token_to_id,
        "dataset_size": dataset_size,
        "seed": SEED}

    torch.save(checkpoint, checkpoint_name)

    print(f"Saved {checkpoint_name}")

print("\nAll MIMIC models trained.")