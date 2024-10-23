import numpy as np
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, TimeDistributed, Dropout
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained model for embedding (BERT)
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Preprocess function to encode experiences (jobs, skills, education)
def preprocess_data(profiles):
    X, y = [], []
    
    for profile in profiles:
        job_titles = profile['job_titles']
        job_skills = profile['skills']

        # Encode job titles and skills using BERT embeddings
        job_embeddings = [bert_model.encode(job) for job in job_titles]
        skill_embeddings = [bert_model.encode(skill) for skill in job_skills]

        # Combine job embeddings and skill embeddings
        profile_seq = [np.concatenate([job_embed, skill_embed]) for job_embed, skill_embed in zip(job_embeddings, skill_embeddings)]
        
        X.append(profile_seq[:-1])  # All but the last job (input)
        y.append(profile_seq[1:])  # All but the first job (target, shifted by 1)
        
    return np.array(X), np.array(y)

# Dummy data (In real cases, this would be extracted from LinkedIn profiles)
profiles = [
    {
        'job_titles': ['Software Engineer', 'Senior Software Engineer', 'Engineering Manager'],
        'skills': ['Python, Java', 'Python, Java, Management', 'Leadership, Management']
    },
    {
        'job_titles': ['Data Analyst', 'Data Scientist', 'Senior Data Scientist'],
        'skills': ['SQL, Excel', 'Python, Machine Learning', 'Deep Learning, AI']
    }
]

# Preprocess profiles to get input and output sequences
X, y = preprocess_data(profiles)

# Pad sequences to ensure equal length for input
max_seq_length = max([len(seq) for seq in X])  # Find the max sequence length
X_padded = pad_sequences(X, maxlen=max_seq_length, dtype='float32', padding='post')
y_padded = pad_sequences(y, maxlen=max_seq_length, dtype='float32', padding='post')

# Define the adjusted LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(None, 768), return_sequences=True))  # Adjusted input shape to 768
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(64, activation='relu')))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(768, activation='linear')))  # Output size is also 768

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_padded, test_size=0.2)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Save the trained model to a file
model.save('lstm_career_model.h5')  # Save as an HDF5 file
print("Model saved as 'lstm_career_model.h5'")

# Load the saved LSTM model
lstm_model = load_model('lstm_career_model.h5')

# Build a job title vocabulary
job_title_vocab = ['Software Engineer', 'Senior Software Engineer', 'Engineering Manager',
                   'Data Analyst', 'Data Scientist', 'Senior Data Scientist', 'Junior Developer',
                   'Software Developer']  # Add more as needed

# Get BERT embeddings for each job title and concatenate with a dummy skill embedding (zeros)
dummy_skill_embedding = np.zeros(384)  # Dummy skill embedding of size 384
job_title_embeddings = {title: np.concatenate([bert_model.encode(title), dummy_skill_embedding]) for title in job_title_vocab}

# Define the predict_next_jobs function
def predict_next_jobs(current_profile, num_predictions=10):
    job_titles = current_profile['job_titles']
    job_skills = current_profile['skills']
    
    # Encode current jobs and skills
    current_embeddings = [np.concatenate([bert_model.encode(job), bert_model.encode(skill)]) 
                          for job, skill in zip(job_titles, job_skills)]
    
    # Pad sequence
    current_embeddings_padded = pad_sequences([current_embeddings], maxlen=max_seq_length, dtype='float32', padding='post')
    
    # Predict next job embeddings
    predicted_job_titles = []
    
    for _ in range(num_predictions):
        predicted_embeddings = lstm_model.predict(current_embeddings_padded)[0]  # Predict the next job embedding
        
        # Compute cosine similarity and find the closest job title
        similarities = {title: cosine_similarity([predicted_embeddings[-1]], [emb])[0][0]
                        for title, emb in job_title_embeddings.items()}
        best_match = max(similarities, key=similarities.get)  # Get the job title with the highest similarity
        predicted_job_titles.append(best_match)
        
        # Add the predicted job back to the input for further predictions
        next_job_embedding = np.concatenate([bert_model.encode(best_match), np.zeros(384)])  # Dummy skill embedding
        current_embeddings_padded = np.concatenate([current_embeddings_padded[:, 1:], [[next_job_embedding]]], axis=1)
    
    return predicted_job_titles

# Example: Predict the next job for a user based on their current experience
test_profile = {
    'job_titles': ['Junior Developer', 'Software Developer'],
    'skills': ['C++, JavaScript', 'Python, JavaScript']
}

# Get up to 10 predicted jobs
predicted_jobs = predict_next_jobs(test_profile, num_predictions=10)
print(predicted_jobs)