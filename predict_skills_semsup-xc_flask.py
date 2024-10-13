"""from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sentence_transformers import SentenceTransformer, util
import requests
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users_jobs.db'
db = SQLAlchemy(app)

# SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    skills = db.Column(db.String(200), nullable=True)  # Store user skills as comma-separated values

# Job Model
class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    scores = db.Column(db.JSON, nullable=True)  # Store similarity scores as JSON

@app.route('/')
def login_register():
    return render_template('login_register.html')

@app.route('/auth', methods=['POST'])
def auth():
    action = request.form.get('action')
    username = request.form.get('username')
    password = request.form.get('password')

    if action == 'Login':
        # Login flow
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password, password):
            return redirect(url_for('login_register'))

        # Log in the user
        session['user_id'] = user.id
        session['username'] = user.username

        # Redirect to skill input if user has no skills
        if not user.skills:
            return redirect(url_for('set_skills'))

        return redirect(url_for('dashboard'))

    elif action == 'Register':
        # Registration flow
        if User.query.filter_by(username=username).first():
            return redirect(url_for('login_register'))  # User already exists

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)

        db.session.add(new_user)
        db.session.commit()

        # Log in the user automatically after registration
        session['user_id'] = new_user.id
        session['username'] = new_user.username

        return redirect(url_for('set_skills'))

@app.route('/set_skills')
def set_skills():
    return render_template('set_skills.html')

@app.route('/set_skills', methods=['POST'])
def set_skills_post():
    skills = request.form.get('skills')
    user_id = session.get('user_id')

    user = User.query.get(user_id)
    user.skills = skills
    db.session.commit()

    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    skills = user.skills.split(',') if user.skills else []

    # Fetch job recommendations
    recommended_jobs = get_job_recommendations(skills)

    # Check for issues with job data
    if not recommended_jobs:
        print("No jobs found or an issue occurred")

    return render_template('dashboard.html', username=user.username, user=user, jobs=recommended_jobs)

@app.route('/update_skills', methods=['POST'])
def update_skills():
    skills = request.form.get('skills')
    user_id = session.get('user_id')

    user = User.query.get(user_id)
    user.skills = skills
    db.session.commit()

    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_register'))

def get_job_recommendations(skills):
    API_URL = "https://jooble.org/api"
    API_KEY = "2ddb2998-39a3-4437-a4e7-1b1fe279b91e"  # Replace with your actual API key
    headers = {"Content-Type": "application/json"}
    body = {"keywords": " ".join(skills), "location": ""}

    try:
        response = requests.post(f'{API_URL}/{API_KEY}', json=body, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors

        jobs = response.json()
        job_descriptions = [job['snippet'] for job in jobs.get('jobs', [])]
        job_titles = [job['title'] for job in jobs.get('jobs', [])]

        skill_embeddings = model.encode(skills, convert_to_tensor=True)
        job_description_embeddings = model.encode(job_descriptions, convert_to_tensor=True)

        all_scores = []
        for job_emb in job_description_embeddings:
            scores = util.pytorch_cos_sim(job_emb, skill_embeddings).squeeze().tolist()
            scores_dict = {skill: score for skill, score in zip(skills, scores)}
            all_scores.append(scores_dict)

        # Clear existing jobs before adding new ones
        Job.query.delete()
        db.session.commit()

        for title, description, scores_dict in zip(job_titles, job_descriptions, all_scores):
            job = Job(title=title, description=description, scores=scores_dict)
            db.session.add(job)
        db.session.commit()

        sorted_jobs = sorted(
            [{'title': title, 'description': desc, 'scores': scores_dict}
             for title, desc, scores_dict in zip(job_titles, job_descriptions, all_scores)],
            key=lambda x: sum(x['scores'].values()) / len(x['scores']),
            reverse=True
        )
        return sorted_jobs
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        return []

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
"""


from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.mutable import MutableDict
from werkzeug.security import generate_password_hash, check_password_hash
from sentence_transformers import SentenceTransformer, util
import requests
import spacy


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)


# SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')


# Load the NER model
ner_model_path = r"C:\Users\saiha\OneDrive\Sai\skill_predict_semsup-xc\model-best"
ner_model = spacy.load(ner_model_path)


# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    skills = db.Column(db.String(200), nullable=True)  # Store user skills as comma-separated values


# Job Model
class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Job linked to a specific user
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    scores = db.Column(MutableDict.as_mutable(db.JSON), nullable=True)  # Store similarity scores as JSON
    interaction_score = db.Column(db.Float, default=1.0)  # New interaction score
    user = db.relationship('User', backref=db.backref('jobs', lazy=True))  # Relationship with the user


# User Job Interaction Model
class UserJobInteraction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=False)
    liked = db.Column(db.Boolean, nullable=True)
    disliked = db.Column(db.Boolean, nullable=True)
    interaction_increased = db.Column(db.Boolean, nullable=False, default=False)  # Track if interaction score increased
    interaction_decreased = db.Column(db.Boolean, nullable=False, default=False)  # Track if interaction score decreased
    user = db.relationship('User', backref=db.backref('job_interactions', lazy=True))
    job = db.relationship('Job', backref=db.backref('user_interactions', lazy=True))


@app.route('/')
def login_register():
    return render_template('login_register.html')


@app.route('/auth', methods=['POST'])
def auth():
    action = request.form.get('action')
    username = request.form.get('username')
    password = request.form.get('password')

    if action == 'Login':
        # Login flow
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password, password):
            return redirect(url_for('login_register'))

        # Log in the user
        session['user_id'] = user.id
        session['username'] = user.username

        # Redirect to skill input if user has no skills
        if not user.skills:
            return redirect(url_for('set_skills'))

        return redirect(url_for('dashboard'))

    elif action == 'Register':
        # Registration flow
        if User.query.filter_by(username=username).first():
            return redirect(url_for('login_register'))  # User already exists

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)

        db.session.add(new_user)
        db.session.commit()

        # Log in the user automatically after registration
        session['user_id'] = new_user.id
        session['username'] = new_user.username

        return redirect(url_for('set_skills'))


@app.route('/set_skills', methods=['GET', 'POST'])
def set_skills():
    if request.method == 'POST':
        personal_description = request.form.get('description')
        user_id = session.get('user_id')

        # Extract skills using NER model
        doc = ner_model(personal_description)
        extracted_skills = [ent.text for ent in doc.ents if ent.label_ == 'SKILLS']

        # Render the page with extracted skills
        return render_template('set_skills.html', extracted_skills=extracted_skills)

    # Render the page without skills if it's a GET request
    return render_template('set_skills.html', extracted_skills=[])


@app.route('/save_skills', methods=['POST'])
def save_skills():
    user_id = session.get('user_id')
    user = User.query.get(user_id)

    # Retrieve selected skills and any newly added skill
    selected_skills = request.form.getlist('selected_skills[]')  # Use array format to handle multiple values
    extracted_skills = request.form.getlist('extracted_skills[]')
    new_skill = request.form.get('new_skill')

    # If there's a new skill, append it to the list
    if new_skill:
        selected_skills.append(new_skill)

    # Remove duplicates (if necessary) and save skills to the database
    all_skills = set(selected_skills + extracted_skills)
    user.skills = ', '.join(all_skills)
    db.session.commit()

    # Delete any old jobs before showing new recommendations
    Job.query.filter_by(user_id=user_id).delete()
    db.session.commit()

    # Redirect to the dashboard after saving skills and deleting old jobs
    return redirect(url_for('dashboard'))


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():

    user_id = session.get('user_id')
    user = User.query.get(user_id)  # Fetch the current logged-in user

    if not user:
        return redirect(url_for('login_register'))  # Redirect to login if no user is found

    skills = user.skills.split(',') if user.skills else []

    if request.method == 'POST':
        # Handling the like/dislike action
        job_id = request.form.get('job_id')
        action = request.form.get('action')
        job = Job.query.get(job_id)

        interaction = UserJobInteraction.query.filter_by(user_id=user_id, job_id=job_id).first()

        if not interaction:
            interaction = UserJobInteraction(user_id=user_id, job_id=job_id)

        if action == 'like':
            interaction.liked = True
            interaction.disliked = False
            job.interaction_score += 0.2  # Increase interaction score for liked jobs
            flash(f"Interaction score for job '{job.title}' increased by 0.2", 'success')

        elif action == 'dislike':
            interaction.liked = False
            interaction.disliked = True
            job.interaction_score -= 0.2  # Decrease interaction score for disliked jobs
            flash(f"Interaction score for job '{job.title}' decreased by 0.2", 'danger')

        db.session.add(interaction)
        db.session.commit()

        # Update interaction scores for related jobs based on the current job's interaction score
        job_titles = [j.title for j in Job.query.filter_by(user_id=user_id).all()]
        job_embeddings = model.encode(job_titles, convert_to_tensor=True)
        current_job_embedding = model.encode(job.title, convert_to_tensor=True)

        cosine_similarities = util.pytorch_cos_sim(current_job_embedding, job_embeddings).squeeze().tolist()

        for idx, score in enumerate(cosine_similarities):
            related_job = Job.query.filter_by(user_id=user_id).all()[idx]
            if related_job.id != job.id and score > 0.7:  # Only affect jobs with a similarity score > 0.7

                interaction = UserJobInteraction.query.filter_by(user_id=user_id, job_id=related_job.id).first()

                if not interaction:
                    interaction = UserJobInteraction(user_id=user_id, job_id=related_job.id)

                if action == 'like':
                    change = 0.1
                    interaction.interaction_increased = True
                    interaction.interaction_decreased = False  # Reset the other flag if needed
                else:
                    change = -0.1
                    interaction.interaction_decreased = True
                    interaction.interaction_increased = False  # Reset the other flag if needed
                    
                related_job.interaction_score += change
                db.session.add(interaction)  # Save the interaction changes
                flash(f"Interaction score for related job '{related_job.title}' changed by {change}", 'info')

                db.session.commit()

    # Fetch job recommendations
    recommended_jobs = get_job_recommendations(skills)

    # Remove duplicate job listings based on job title and company
    unique_jobs = {}
    for job in recommended_jobs:
        # Use a combination of title and company as a unique identifier
        identifier = (job.title, job.description)
        if identifier not in unique_jobs:
            unique_jobs[identifier] = job

    for job in recommended_jobs:
        print(f"Job Title: {job.title}")
        print(f"Scores Structure: {job.scores}")

    increased_interaction_jobs = increased_interaction_jobs = Job.query.join(UserJobInteraction).filter(
        UserJobInteraction.user_id == user_id,
        UserJobInteraction.interaction_decreased == False,
        UserJobInteraction.interaction_increased == True
    ).all()

    decreased_interaction_jobs = decreased_interaction_jobs = Job.query.join(UserJobInteraction).filter(
        UserJobInteraction.user_id == user_id,
        UserJobInteraction.interaction_decreased == True,
        UserJobInteraction.interaction_increased == False
    ).all()
    
    # Pass the user object and job recommendations to the template
    return render_template(
        'dashboard.html', 
        jobs=recommended_jobs, 
        increased_jobs=increased_interaction_jobs, 
        decreased_jobs=decreased_interaction_jobs
    )

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_register'))


def calculate_similarity(job_scores, reference_scores):
    """Calculate similarity between job scores and reference scores."""
    scores = list(job_scores.values())
    ref_scores = list(reference_scores.values())
    if not scores or not ref_scores or sum(ref_scores) == 0:
        return 0
    return sum(s * r for s, r in zip(scores, ref_scores)) / (sum(ref_scores) * len(scores))


def get_job_recommendations(skills):
    API_URL = "https://jooble.org/api"
    API_KEY = "2ddb2998-39a3-4437-a4e7-1b1fe279b91e"
    headers = {"Content-Type": "application/json"}
    body = {"keywords": " ".join(skills), "location": "", "page": 1}

    user_id = session.get('user_id')
    if not user_id:
        return []  # No user logged in

    jobs = []
    try:
        for page in range(1, 2):  # Example: Make 20 requests to fetch jobs across 20 pages
            body['page'] = page
            response = requests.post(f'{API_URL}/{API_KEY}', json=body, headers=headers)
            response.raise_for_status()
            data = response.json()
            jobs.extend(data.get('jobs', []))

            if len(data.get('jobs', [])) == 0:
                break  # Stop if no more jobs are returned

        job_descriptions = [job['snippet'] for job in jobs]

        for i, description in enumerate(job_descriptions):
            job_title = jobs[i]['title']

            # Check if this job already exists for this user
            existing_job = Job.query.filter_by(user_id=user_id, title=job_title, description=description).first()
            if existing_job:
                continue  # Skip saving duplicate jobs

            job_scores = {}

            # Compute skill-based similarity scores
            for skill in skills:
                title_embedding = model.encode(job_title)
                description_embedding = model.encode(description)
                skill_embedding = model.encode(skill)

                # Title and description similarity scores
                title_similarity = util.cos_sim(title_embedding, skill_embedding).item()
                description_similarity = util.cos_sim(description_embedding, skill_embedding).item()

                # Weighted score calculation: 70% title, 30% description
                job_scores[skill] = 0.7 * title_similarity + 0.3 * description_similarity

            # Save job to the database for this user
            new_job = Job(
                user_id=user_id,
                title=job_title,
                description=description,
                scores=job_scores,
                interaction_score=1.0
            )

            db.session.add(new_job)
            db.session.commit()

        # Fetch all jobs for the current user after scoring and saving
        saved_jobs = Job.query.filter_by(user_id=user_id).all()

        # Sort jobs based on the dynamically calculated weighted similarity
        sorted_jobs = sorted(
            saved_jobs,
            key=lambda job: sum(0.7 * util.cos_sim(model.encode(job.title), model.encode(skill)).item() +
                                0.3 * util.cos_sim(model.encode(job.description), model.encode(skill)).item()
                                for skill in skills),
            reverse=True
        )
        return sorted_jobs

    except requests.RequestException as e:
        print(f"Error fetching jobs: {e}")
        return []
    

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)