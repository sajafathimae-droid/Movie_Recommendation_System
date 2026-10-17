
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

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

vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(df["genres"])
similarity = cosine_similarity(genre_matrix)


@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    selected_movie = ""

    if request.method == "POST":
        selected_movie = request.form.get("movie", "").strip()

        matches = df.index[
            df["title"].str.lower() == selected_movie.lower()
        ]

        if len(matches) > 0:
            movie_index = matches[0]

            scores = list(enumerate(similarity[movie_index]))
            scores.sort(key=lambda item: item[1], reverse=True)

            for index, score in scores:
                if index != movie_index:
                    recommendations.append({
                        "title": df.iloc[index]["title"],
                        "similarity": round(float(score) * 100, 1)
                    })

                if len(recommendations) == 5:
                    break

    return render_template(
        "index.html",
        movies=df["title"].tolist(),
        recommendations=recommendations,
        selected_movie=selected_movie
    )


if __name__ == "__main__":
    app.run(debug=True)