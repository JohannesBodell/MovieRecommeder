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

# Parameters
embedding_size = 20
learning_rate = 0.01
epochs = 15
batch_size = 256

ratings = load_ratings()

user_to_index, movie_to_index = create_mappings(ratings)


# Split the data into training, validation, and test sets
train_ratings, temp_ratings = train_test_split(
    ratings,
    test_size=0.3,
    random_state=42
)

validation_ratings, test_ratings = train_test_split(
    temp_ratings,
    test_size=0.5,
    random_state=42
)



train_dataset = RatingsDataset(
    train_ratings,
    user_to_index,
    movie_to_index
)

validation_dataset = RatingsDataset(
    validation_ratings,
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
    batch_size=batch_size,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

model = MatrixFactorization(
    num_users=len(user_to_index),
    num_movies=len(movie_to_index),
    embedding_size=embedding_size
)

loss_function = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

global_mean = train_ratings['rating'].mean()

for epoch in range(epochs):
    model.train()

    total_loss = 0

    for users, movies, ratings in train_loader:
        optimizer.zero_grad()
        predictions = model(users, movies)
        loss = loss_function(predictions, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    metrics = evaluate_model(model, validation_loader)
    print(f"Epoch {epoch + 1}"
          f", Total Loss: {total_loss/len(train_loader):.4f}"
          f", Validation MAE: {metrics['mae']:.4f}"
          f", Validation RMSE: {metrics['rmse']:.4f}")
    
    evaluation_results = evaluate_model(model, test_loader)
