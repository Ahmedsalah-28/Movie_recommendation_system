from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load precomputed data and models
movies = pickle.load(open('artificats/movies.pkl', 'rb'))
knn = pickle.load(open('artificats/knn_model.pkl', 'rb'))
csr_data = pickle.load(open('artificats/csr_data.pkl', 'rb'))
final_dataset = pickle.load(open('artificats/final_dataset.pkl', 'rb'))

def get_recommendation(movie_name):
    movie_list = movies[movies['title'].str.contains(movie_name, case=False)]
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distance, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=11)
        rec_movies_indices = sorted(list(zip(indices.squeeze().tolist(), distance.squeeze().tolist())), key=lambda x: x[1])[:0: -1]
        recommended_movies = []
        for val in rec_movies_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommended_movies.append(movies.iloc[idx]['title'].values[0])
        return recommended_movies
    else:
        return ["Movie not found. Please try another title."]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    movie = request.form.get('movie')
    recommendations = get_recommendation(movie)
    return render_template('recommendations.html', movie=movie, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
