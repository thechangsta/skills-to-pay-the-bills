import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd

df = pd.read_pickle('./skill.pickle')

# load BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# put model on GPU (is done in-place)
model.to(device)

# function to get word embeddings
def get_BERT_embedding(text):
    # tokenize input
    encoding = tokenizer(text,
                       return_tensors='pt',
                       padding=True,
                       truncation=True)
    input_ids = encoding['input_ids'].to(device)
    mask = encoding['attention_mask'].to(device)
    # generate embedding
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=mask)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

def magnitute(vec):
    return np.sqrt(np.sum(vec**2))

df['final_skill_embedding_mag'] = df['final_skill_embedding'].apply(magnitute)

def rank_jobs(my_skills):
    emb1 = get_BERT_embedding(my_skills).T
    mag_emb = np.sqrt(np.sum(emb1**2))
    res = []
    print(device)
    for index, row in tqdm(df.iterrows(), desc="Calc Similarity"):
        res.append(np.dot(row['final_skill_embedding'], emb1)/(mag_emb*row['final_skill_embedding_mag']))
    sorted_indices = sorted(range(len(res)), key=lambda i: res[i], reverse=True)
    return sorted_indices

def getRecs(my_skills):
    sorted_indices = rank_jobs(my_skills)
    jobs = []
    for i in sorted_indices[:10]:
        jobs.append(df['job_title'][i])
#     jobs
    top_skills = {}
    for i in sorted_indices[:10000]:
        for skill in df['job_skills'][i].split(","):
            skill = skill.strip().lower()
            if skill not in top_skills:
                top_skills[skill] = 1
            else:
                top_skills[skill] += 1
    sorted_top_skills = dict(sorted(top_skills.items(), key=lambda item: item[1], reverse=True))
    for userskill in my_skills.split(","):
        sorted_top_skills.pop(userskill, None)
    skills = list(dict(list(sorted_top_skills.items())[:10]).keys())
    
    return {"job_titles": jobs, "skills": skills}



# Import necessary packages
from flask import Flask, request, jsonify
from pyngrok import ngrok

# Create a Flask app
app = Flask(__name__)

query_cache = {}

# Define a function that processes the provided skills and returns a response
def process_skills(skills):
    # Convert the skills string into a list of skills
    skills_list = skills['skills'].split(",")
    
    # Process the skills list as needed
    # For demonstration purposes, let's return the number of skills
    num_skills = len(skills_list)
    
    # Extract the skills from the JSON data
    skills = []
    for skill in skills_list:
        skills.append(skill.strip().lower())
    
    
    # Return a JSON response with the processed data
#     response_data = {
#         "message": "Skills processed successfully.",
#         "number_of_skills": num_skills,
#         "skills": skills
#     }
    skills = ",".join(skills)
    print(skills)
    if skills in query_cache:
        # Key exists, return its value
        response_data = query_cache[skills]
    else:
        response_data = getRecs(skills)
        query_cache[skills] = response_data
    return response_data

# Define a route for the API
@app.route('/api/process-skills', methods=['POST'])
def api_process_skills():
    # Get the JSON data from the request
    data = request.get_json()
    
    # Process the skills using the function
    response_data = process_skills(data)
    
    # Return the response data as JSON
    return jsonify(response_data)


# Start ngrok and expose the Flask app
ngrok_tunnel = ngrok.connect(5001)
print(f"Flask app is running! Access it at: {ngrok_tunnel.public_url}/api/process-skills")

# Print the ngrok URL so you can use it to access the Flask app remotely


app.run(host='0.0.0.0', port=5001, use_reloader=False, debug=True)