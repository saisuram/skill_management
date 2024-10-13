import spacy

# insert job description
job_description = """
I got my bachelors degree from stanford in CS and I currently work in Google as a software developer where I work with databases using SQL and Tableau. I also work with the data science team by using Python and R to come up with data visualizations and data science/ML algorithms
"""


ner_model_path = r"C:\Users\saiha\OneDrive\Sai\skill_predict_semsup-xc\model-best"

ner_model = spacy.load(ner_model_path)

# test the algorithm
doc = ner_model(job_description)

list1 = []

for ent in doc.ents:
    if ent.label_ == "SKILLS" and ent.text not in list1:
        list1.append(ent.text)
print(list1)