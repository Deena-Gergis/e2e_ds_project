TRACKING_URI = "file:///home/deena_gergis/iti/iti_e2e_live/notebooks/mlruns/"
EXPERIMENT_ID = "1"
RUN_ID = "4c35a9c9d26a48e8bfccfae05a63d348"
CLUSTERS_YAML_PATH = "/home/deena_gergis/iti/iti_e2e_live/data/processed/features_skills_clusters_description.yaml"

#------------------------------------------

import JobPrediction
from JobPrediction import JobPrediction

import pandas as pd
from flask import Flask, request, jsonify

#------------------------------------------

# Initiate API and JobPrediction object
app = Flask(__name__)
job_model = JobPrediction(tracking_uri=TRACKING_URI, 
                          experiment_id=EXPERIMENT_ID, 
                          run_id=RUN_ID, 
                          clusters_yaml_path=CLUSTERS_YAML_PATH)


# Create prediction endpoint 
@app.route('/predict_jobs_probs', methods=['POST'])
def predict_jobs_probs():
    available_skills = request.get_json()
    predictions = job_model.predict_jobs_probabilities(available_skills).to_dict()    
    return jsonify(predictions)


if __name__ == '__main__':
    app.run()
    