import torch
import torch.nn as nn
from data_synthea import vocab, encode, patient_to_tokens, patients

# Initial transformer configuration
VOCAB_SIZE = len(vocab)
EMBED_DIM = 32
NUM_HEADS = 2
CONTEXT_LENGTH = 128


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = EMBED_DIM // NUM_HEADS

        # randomly initialize a weight matrix that transforms 32 dimensions into 16
        # Head 1
        self.q1 = nn.Linear(EMBED_DIM, self.head_dim)
        self.k1 = nn.Linear(EMBED_DIM, self.head_dim)
        self.v1 = nn.Linear(EMBED_DIM, self.head_dim)

        # Head 2
        self.q2 = nn.Linear(EMBED_DIM, self.head_dim)
        self.k2 = nn.Linear(EMBED_DIM, self.head_dim)
        self.v2 = nn.Linear(EMBED_DIM, self.head_dim)

        # Combine the two heads' outputs in the 32-dimensional model space
        self.output = nn.Linear(EMBED_DIM, EMBED_DIM)

    def forward(self, x):
        # Input shape: [B, T, 32].
        B, T, C = x.shape

        # Generate Q, K, and V independently for each head.
        # performs the actual matrix multiplication and there's a bias vector
        # Every projection produces [B, T, 16]
        # Q_1 = X*W_Q1) there's a bias vector added too
        q1 = self.q1(x)
        k1 = self.k1(x)
        v1 = self.v1(x)

        q2 = self.q2(x)
        k2 = self.k2(x)
        v2 = self.v2(x)

        # Calculate scaled attention scores for each head using tranbsposed key matrix
        # the transpose fucniton requires specifiyign whihc dimension to transpose and its 
        # B,T,16 and T and 16 ar eteh ones we want to transpose
        # Each produces a [B, T, T] attention-score matrix.
        scores1 = q1 @ k1.transpose(1, 2) / (self.head_dim ** 0.5)
        scores2 = q2 @ k2.transpose(1, 2) / (self.head_dim ** 0.5)

        # Apply the same causal mask to both heads.
        # Tokens can attend to themselves and previous tokens, not future ones.
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores1 = scores1.masked_fill(~mask, float("-inf"))
        scores2 = scores2.masked_fill(~mask, float("-inf"))

        # Convert scores into attention weights.
        # Each token's allowed attention weights sum to 1.
        weights1 = torch.softmax(scores1, dim=-1)
        weights2 = torch.softmax(scores2, dim=-1)

        # Apply attention weights to the value vectors.
        # Each head produces contextualized vectors of shape [B, T, 16].
        head1_output = weights1 @ v1
        head2_output = weights2 @ v2

        # Concatenate the two heads along the feature dimension.
        # [B, T, 16] + [B, T, 16] -> [B, T, 32].
        combined = torch.cat([head1_output, head2_output], dim=-1)

        # Apply the learned output projection.
        # Mix information from both heads while retaining 32 dimensions.
        output = self.output(combined)

        return output


class PatientTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = CausalSelfAttention()
        # Normalize each token's 32-dimensional representation
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        # Feed-forward neural network: 32 -> 64 -> 32.
        # GELU introduces nonlinearity between the two linear layers
        self.feed_forward = nn.Sequential(
            nn.Linear(EMBED_DIM, 64),
            nn.GELU(),
            nn.Linear(64, EMBED_DIM)
        )

        # Final normalization before next-token prediction
        self.final_norm = nn.LayerNorm(EMBED_DIM)
        # Convert each 32-dimensional vector into vocabulary scores
        # 
        self.output_layer = nn.Linear(EMBED_DIM, VOCAB_SIZE)


        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        """
        The positional embedding retains sequence, the order of the input token list 
        is not captured by attention heads in token embedding so the positinal relationships
        of the tokens themselves are actually learned, so not just diabetes, but diabetes_pos_2 vs diabetes_pos_5
        There is ONE shared positional embedding matrix for the entire model,
        The same token receives the same token embedding regardless of where it appears. Likewise, position 2 
        always receives the same positional embedding, regardless of which token occupies it.
        But when we add them together, we create a representation of that particular token at that particular position 
        diabetes_pos_i vs diabetes_pos_j and we want to learn how to treat them differently
        """
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, EMBED_DIM)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape

        # Convert token IDs into 32-dimensional vectors
        token_vectors = self.token_embedding(token_ids)

        # Create positional vectors for each sequence position
        positions = torch.arange(seq_len, device=token_ids.device) # position indices
        position_vectors = self.position_embedding(positions)

        # Combine token identity and position

        # Combine token identity and positional information.
        # Shape: [B, T, 32].
        x = token_vectors + position_vectors

        # Normalize, apply attention, and add the original input.
        # The residual connection preserves information from before attention
        attention_input = self.norm1(x)
        attention_output = self.attention(attention_input)
        # residual connection
        x = x + attention_output

        # Normalize and process each token through the feed-forward NN.
        # The NN expands each vector from 32 to 64, then reduces it to 32.
        feed_forward_input = self.norm2(x)
        feed_forward_output = self.feed_forward(feed_forward_input)

        # Add the second residual connection
        x = x + feed_forward_output
        #  Normalize the final token representations.
        x = self.final_norm(x)
        # Produce a score (logit) for every vocabulary token.
        # Shape: [B, T, 32] -> [B, T, 10].
        logits = self.output_layer(x)

        return logits



if __name__ == "__main__":
    # we have hard coded the dims in the self_init so it sets up the randomzed embeddings automatically
    # Creates and stores two randomly initialized matrices: Token embeddings: 10 tokens × 32 dimensions.
    # Position embeddings: 16 positions × 32 dimensions, these basically serve as look up tables
    model = PatientTransformer()

    # Encode one patient and add a batch dimension
    tokens = patient_to_tokens(patients[0])
    ids = encode(tokens)
    x = torch.tensor([ids], dtype=torch.long)

    # this actually executes the forward pass automatically as a property of nn.Module, uses _call_() 
    # forward pass is limited to converting one patient's sequence of integer IDs into a sequence of 
    # numerical vectors taken from the model, containing token and positional information.
    output = model(x)

    print("Input shape:", x.shape)
    print("Model output shape:", output.shape)
    print("Embedding dimension:", EMBED_DIM)