import torch
import torch.nn as nn
import torch.optim as optim

# Define the custom embedding layer
class MyEmbeddingLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(MyEmbeddingLayer, self).__init__()
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, input):
        return self.embeddings[input]

# Assume the indices of the words we're interested in are [0, 1, 2]
word_to_index = {'food': 0, 'animal': 1, 'human': 2}

# Assume some training data (pairs of indices of words that appear near each other)
positive_samples = [(0, 1), (1, 2), (2, 0)]  # Example pairs of words that appear near each other
negative_samples = [(0, 2), (1, 0), (2, 1)]  # Example pairs of words that do not appear near each other

# Create the embedding layer
embedding_layer = MyEmbeddingLayer(num_embeddings=10, embedding_dim=3)

# Define the loss function and optimizer
optimizer = optim.SGD(embedding_layer.parameters(), lr=0.1)

# Train the embedding layer using negative sampling
for epoch in range(100):
    total_loss = 0
    for (pos_u, pos_v), (neg_u, neg_v) in zip(positive_samples, negative_samples):
        optimizer.zero_grad()

        # Positive sample loss
        pos_u_embedding = embedding_layer(torch.tensor(pos_u))
        pos_v_embedding = embedding_layer(torch.tensor(pos_v))
        pos_score = torch.dot(pos_u_embedding, pos_v_embedding)
        pos_loss = -torch.log(torch.sigmoid(pos_score))

        # Negative sample loss
        neg_u_embedding = embedding_layer(torch.tensor(neg_u))
        neg_v_embedding = embedding_layer(torch.tensor(neg_v))
        neg_score = torch.dot(neg_u_embedding, neg_v_embedding)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_score))

        # Total loss
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {total_loss}')

# Function to compute cosine similarity
def cosine_similarity(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

# Compare the words using the trained embeddings
word_embeddings = embedding_layer(torch.tensor([0, 1, 2]))
similarity_food_animal = cosine_similarity(word_embeddings[0], word_embeddings[1])
similarity_food_human = cosine_similarity(word_embeddings[0], word_embeddings[2])
similarity_animal_human = cosine_similarity(word_embeddings[1], word_embeddings[2])

print(f"Similarity between food and animal: {similarity_food_animal.item():.2f}")
print(f"Similarity between food and human: {similarity_food_human.item():.2f}")
print(f"Similarity between animal and human: {similarity_animal_human.item():.2f}")
