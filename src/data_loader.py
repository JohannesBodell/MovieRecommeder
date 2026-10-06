import pandas as pd

def load_movies():
    return pd.read_csv("data/movies.csv")

def load_ratings():
    return pd.read_csv("data/ratings.csv")




if __name__ == "__main__":
    movies = load_movies()
    ratings = load_ratings()

    print(movies.head())
    print()
    print(ratings.head())