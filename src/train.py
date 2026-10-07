from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
from torch import nn
from models.matrix_factorization import MatrixFactorization
from evaluation.metrics import evaluate_model
from data_loader import load_ratings
from datasets.ratings_dataset import (
    RatingsDataset,
    create_mappings
)


# 1. Läs in alla ratings
ratings = load_ratings()


# 2. Skapa våra ID -> index mappings
user_to_index, movie_to_index = create_mappings(ratings)


# 3. Dela upp datan
train_ratings, test_ratings = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)


# 4. Skapa PyTorch datasets
train_dataset = RatingsDataset(
    train_ratings,
    user_to_index,
    movie_to_index
)

test_dataset = RatingsDataset(
    test_ratings,
    user_to_index,
    movie_to_index
)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False
)

model = MatrixFactorization(
    num_users=len(user_to_index),
    num_movies=len(movie_to_index),
    embedding_size=20
)

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    model.train()

    total_loss = 0

    for users, movies, ratings in train_loader:
        optimizer.zero_grad()
        predictions = model(users, movies)
        loss = loss_function(predictions, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    metrics = evaluate_model(model, test_loader)
    print(f"Epoch {epoch + 1}"
          f", Total Loss: {total_loss/len(train_loader):.4f}"
          f", Test MAE: {metrics['mae']:.4f}"
          f", Test RMSE: {metrics['rmse']:.4f}")
    
    evaluation_results = evaluate_model(model, test_loader)
