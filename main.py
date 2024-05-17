from flask import Flask, request, jsonify
import spacy
from rake_nltk import Rake
import random
import google.generativeai as genai
from flask_cors import CORS

# Download model
import nltk
import ssl
# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')
nltk.download('stopwords')

# Configure the generative model
genai.configure(api_key="AIzaSyDUaZrbMBXDgmj8LyMoYq6Ts2pl-j6zsvQ")
model = genai.GenerativeModel('gemini-pro')

app = Flask(__name__)
cors = CORS(app, resource={r"/*": {"origins": "*"}}, methods=['POST', 'GET', 'OPTIONS', 'PUT', 'DELETE'])

# Load spaCy's pre-trained model
nlp = spacy.load("en_core_web_sm")

# Function to extract detailed interests using NER and keyword extraction
def extract_interests(search_queries):
    detailed_interests = []

    for query in search_queries:
        doc = nlp(query)
        # Extract named entities
        entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART"]]
        detailed_interests.extend(entities)
        
        # Extract keywords using RAKE
        rake = Rake()
        rake.extract_keywords_from_text(query)
        keywords = rake.get_ranked_phrases()
        detailed_interests.extend(keywords)

    return detailed_interests

def generate_description(topic):
    response = model.generate_content("Generate a short description that I can use to draw a picture. The picture should be funny and make me relax. The topic is " + topic)
    
    # Debugging: Print the response to understand its structure
    print(response)
    
    # Extract the text from the response based on the correct attribute
    # Adjust this based on the actual response structure
    if hasattr(response, 'text'):
        return response.text
    elif hasattr(response, 'choices'):
        return response.choices[0].text
    else:
        return "Could not generate description"

@app.route('/process_search_history', methods=['POST'])
def process_search_history():
    data = request.json
    search_history = data.get('search_history', [])
    if not search_history:
        return jsonify({"error": "No search history provided"}), 400

    # Extract detailed interests
    detailed_interests = extract_interests(search_history)

    # Ensure unique interests
    unique_interests = list(set(detailed_interests))
    
    if not unique_interests:
        return jsonify({"error": "No unique interests found"}), 400

    # Separate person names and other entities from other topics
    prioritized_topics = [topic for topic in unique_interests if any(ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART"] for ent in nlp(topic).ents)]
    other_topics = [topic for topic in unique_interests if topic not in prioritized_topics]

    # Randomly select three topics, prioritizing person names or places if available
    random_three_topics = []
    if prioritized_topics:
        random_three_topics.extend(random.sample(prioritized_topics, min(3, len(prioritized_topics))))
    remaining_topics_needed = 3 - len(random_three_topics)
    if remaining_topics_needed > 0 and other_topics:
        random_three_topics.extend(random.sample(other_topics, min(remaining_topics_needed, len(other_topics))))

    # Convert the list to a comma-separated string
    random_three_topics_string = ", ".join(random_three_topics)
    
    # Generate a description for the selected topics
    description = generate_description(random_three_topics_string)

    return jsonify({
        "random_three_topics": random_three_topics_string,
        "description": description,
        "detailed_interests": unique_interests
    })

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
