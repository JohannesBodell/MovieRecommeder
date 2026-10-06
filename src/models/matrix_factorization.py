import torch
from torch import nn


class MatrixFactorization(nn.Module):

    def __init__(
        self,
        num_users,
        num_movies,
        embedding_size=20
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_size
        )

        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_size
        )

    def forward(self, user, movie):

        user_vector = self.user_embedding(user)
        movie_vector = self.movie_embedding(movie)

        prediction = (
            user_vector * movie_vector
        ).sum(dim=1)

        return prediction

    @staticmethod
    def create_mappings(ratings):
        unique_users = ratings["userId"].unique()
        unique_movies = ratings["movieId"].unique()

        user_to_index = {
            user_id: index
            for index, user_id in enumerate(unique_users)
        }

        movie_to_index = {
            movie_id: index
            for index, movie_id in enumerate(unique_movies)
        }

        return user_to_index, movie_to_index