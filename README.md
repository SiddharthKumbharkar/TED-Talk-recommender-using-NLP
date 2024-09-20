# TED-Talk-recommender-using-NLP
Made this TED talk recommender to practice NLP. It uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to extract important features from TED Talk summaries and provides recommendations based on user preferences.

Features
Content-based Recommendations: Suggest TED Talks similar to the one you're interested in.
Summaries-based Recommendations: Enter a summary or text to get recommended TED Talks related to the given content.
Custom Search: Recommend TED Talks based on user inputted content, enabling flexible discovery.
TF-IDF Vectorization: Utilizes TF-IDF to represent text data for similarity measurement.
Technologies Used
Natural Language Processing (NLP) with Python
TF-IDF for text vectorization
Cosine Similarity to find content similarity between TED Talk summaries
Pandas for data handling
Scikit-learn for vectorization and similarity computation
How It Works
Preprocessing: TED Talk summaries are preprocessed (tokenized, cleaned) and converted into TF-IDF vectors.
Recommendation Engine: Based on a TED Talk's summary, the engine calculates similarity scores using cosine similarity to recommend talks with the most similar content.
User Input Mode: Users can also input custom text, and the system will recommend TED Talks closely aligned with that text.
Dataset
The dataset contains TED Talk summaries and other related metadata. This project focuses on the summaries to generate recommendations, omitting titles for simplicity.
