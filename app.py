from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Download stopwords
nltk.download('stopwords')

# Load the saved model
with open('model_files.pkl', 'rb') as file:
    model_data = pickle.load(file)

vectorizer = model_data['vectorizer']
cosine_sim = model_data['cosine_sim']
ted = model_data['ted']
vocab = model_data['vocab']  # Load the vocabulary

ps = PorterStemmer()

# Preprocess the user input
def preprocess_input(user_input):
    user_input = re.sub('[^a-zA-Z]', ' ', user_input)
    user_input = user_input.lower()
    user_input = user_input.split()
    user_input = [ps.stem(word) for word in user_input if word not in set(stopwords.words('english'))]
    user_input = ' '.join(user_input)
    return user_input

# Recommendation function
def recommend_ttalk(user_input, n=5):
    user_input_processed = preprocess_input(user_input)
    
    # Create a new vectorizer with the same vocabulary
    new_vectorizer = TfidfVectorizer(vocabulary=vocab, stop_words='english')
    
    # Transform the input using the new vectorizer
    user_input_vec = new_vectorizer.fit_transform([user_input_processed])
    
    # Compute cosine similarity between input and TED talks
    user_sim = cosine_similarity(user_input_vec, cosine_sim).flatten()
    
    # Get indices of the most similar talks
    sim_indices = user_sim.argsort()[-n:][::-1]
    
    # Return top N recommendations as a list of dictionaries
    return ted.iloc[sim_indices][['main_speaker', 'details']].to_dict(orient='records')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    topic = request.form['topic']
    recommendations = recommend_ttalk(topic)
    
    return render_template('recommendations.html', topic=topic, recommendations=recommendations)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
