import pandas as pd

from src.content_recommender import (
    build_user_profile,
    score_movie,
    recommend_movies,
)


def create_movies():
    return pd.DataFrame([
        {
            "movieId": 1,
            "title": "Space Action",
            "genres": "Sci-Fi|Action",
        },
        {
            "movieId": 2,
            "title": "Another Space Movie",
            "genres": "Sci-Fi|Action",
        },
        {
            "movieId": 3,
            "title": "Love Story",
            "genres": "Romance|Drama",
        },
        {
            "movieId": 4,
            "title": "Crime Movie",
            "genres": "Crime|Drama",
        },
    ])


def create_ratings():
    return pd.DataFrame([
        {
            "userId": 1,
            "movieId": 1,
            "rating": 5.0,
        },
        {
            "userId": 1,
            "movieId": 3,
            "rating": 2.0,
        },
        {
            "userId": 2,
            "movieId": 4,
            "rating": 5.0,
        },
    ])

def test_recommend_movies_returns_non_empty_list():
    movies = create_movies()
    ratings = create_ratings()
    result = recommend_movies(1, movies, ratings)
    assert len(result) > 0

def test_recommend_movies_handles_users_with_no_ratings():
    movies = create_movies()
    ratings = create_ratings()
    result = recommend_movies(3, movies, ratings)
    assert len(result) > 0

def test_build_user_profile():
    movies = create_movies()
    ratings = create_ratings()

    profile = build_user_profile(
        userid=1,
        movies=movies,
        ratings=ratings,
    )

    assert profile["Sci-Fi"] == 5.0
    assert profile["Action"] == 5.0
    assert profile["Romance"] == 2.0
    assert profile["Drama"] == 2.0

def test_score_movie():
    profile = {
        "Sci-Fi": 5.0,
        "Action": 4.0,
        "Romance": 2.0,
    }

    movie = {
        "movieId": 100,
        "title": "Test Movie",
        "genres": "Sci-Fi|Action",
    }

    score = score_movie(movie, profile)

    assert score == 4.5

def test_recommend_movies_excludes_seen_movies():
    movies = create_movies()
    ratings = create_ratings()

    recommendations = recommend_movies(
        userid=1,
        movies=movies,
        ratings=ratings,
        limit=10,
    )

    recommended_ids = recommendations["movieId"].tolist()

    assert 1 not in recommended_ids
    assert 3 not in recommended_ids

def test_recommend_movies_prefers_matching_genres():
    movies = create_movies()
    ratings = create_ratings()

    recommendations = recommend_movies(
        userid=1,
        movies=movies,
        ratings=ratings,
        limit=10,
    )

    assert recommendations.iloc[0]["movieId"] == 2