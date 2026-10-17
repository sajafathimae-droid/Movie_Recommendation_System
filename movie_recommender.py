
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
movies = {
    "title": [
        "Harry Potter",
        "Fantastic Beasts",
        "The Lord of the Rings",
        "The Hobbit",
        "Avengers",
        "Iron Man",
        "Finding Nemo",
        "Toy Story",
        "Interstellar",
        "The Martian"
    ],
    "genres": [
        "fantasy magic adventure",
        "fantasy magic adventure",
        "fantasy adventure action",
        "fantasy adventure action",
        "action superhero adventure",
        "action superhero science fiction",
        "animation family adventure",
        "animation family comedy",
        "science fiction space drama",
        "science fiction space adventure"
    ]
}

df = pd.DataFrame(movies)

# Convert movie genres into numerical features
vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(df["genres"])

# Calculate similarity between movies
similarity = cosine_similarity(genre_matrix)


def recommend_movies(movie_title, number=5):
    matches = df.index[df["title"].str.lower() == movie_title.lower()]

    if len(matches) == 0:
        print("Movie not found. Please check the title.")
        return

    movie_index = matches[0]
    scores = list(enumerate(similarity[movie_index]))

    # Sort by similarity and exclude the selected movie
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    scores = [item for item in scores if item[0] != movie_index]

    print(f"\nMovies similar to {df.iloc[movie_index]['title']}:\n")

    for index, score in scores[:number]:
        print(f"{df.iloc[index]['title']} (similarity: {score:.2f})")


# Test the recommendation system
recommend_movies("Harry Potter")