import torch
from torch.utils.data import Dataset


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

class RatingsDataset(Dataset):

    def __init__(
        self,
        ratings,
        user_to_index,
        movie_to_index
    ):
        self.users = [
            user_to_index[x]
            for x in ratings["userId"]
        ]

        self.movies = [
            movie_to_index[x]
            for x in ratings["movieId"]
        ]

        self.ratings = ratings["rating"].tolist()

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, index):

        return (
            torch.tensor(self.users[index]),
            torch.tensor(self.movies[index]),
            torch.tensor(
                self.ratings[index],
                dtype=torch.float32
            )
        )