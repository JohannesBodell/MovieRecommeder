def build_user_profile(userid, movies, ratings):
    user_ratings = ratings[
        ratings["userId"] == userid
    ]

    user_movies = user_ratings.merge(
        movies,
        on="movieId"
    )

    user_genres = user_movies.copy()
    user_genres["genres"] = (
        user_genres["genres"]
        .str.split("|")
    )
    user_genres = user_genres.explode("genres")

    genre_profile = (
        user_genres
        .groupby("genres")["rating"]
        .mean()
        .sort_values(ascending=False)
    )

    return genre_profile

def score_movie(movie, user_profile):
    movie_genres = movie["genres"].split("|")
    score = 0
    for genre in movie_genres:
        if genre in user_profile:
            score += user_profile[genre]
    score = score / len(movie_genres) if movie_genres else 0
    return score

def recommend_movies(userid, movies, ratings, limit=10):
    user_profile = build_user_profile(userid, movies, ratings)
    seen_movies = ratings[ratings["userId"] == userid]["movieId"].tolist()
    movies["score"] = movies.apply(lambda x: score_movie(x, user_profile), axis=1)
    recommended = movies[~movies["movieId"].isin(seen_movies)]
    recommended = recommended.sort_values(by="score", ascending=False).head(limit)
    return recommended