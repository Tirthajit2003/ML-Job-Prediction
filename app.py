from flask import Flask, render_template, send_from_directory, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('job_file.pkl')

# Load your DataFrame
df = pd.read_csv('jobs_in_data_2024.csv')
df.drop_duplicates(inplace=True)

# Get the column names from the DataFrame
columns = df.columns.tolist()

@app.route('/')
def index():
    return render_template('index.html', columns=columns)


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)


@app.route('/predict', methods=['POST'])
def predict_salary():
    # Get input data from the request
    data = request.json
    print(data)

    # Initialize the dictionary with default values
    res = {
        'company_size_L': False,
        'company_size_M': False,
        'company_size_S': False,
        'experience_level_Entry-level': False,
        'experience_level_Executive': False,
        'experience_level_Mid-level': False,
        'experience_level_Senior': False,
        'employment_type_Contract': False,
        'employment_type_Freelance': False,
        'employment_type_Full-time': False,
        'employment_type_Part-time': False,
        'work_setting_Hybrid': False,
        'work_setting_In-person': False,
        'work_setting_Remote': False,
        'job_title_encoded': 0.0,
        'job_category_encoded': 0.0,
        'company_location_encoded': 0.0
    }

    # Set values for company size
    if data['company_size'] == 'L':
        res['company_size_L'] = True
    elif data['company_size'] == 'M':
        res['company_size_M'] = True
    else:
        res['company_size_S'] = True

    # Set values based on user input
    if data['experience_level'] == 'Senior':
        res['experience_level_Senior'] = True
    elif data['experience_level'] == 'Entry-level':
        res['experience_level_Entry-level'] = True
    elif data['experience_level'] == 'Executive':
        res['experience_level_Executive'] = True
    else:
        res['experience_level_Mid-level'] = True

    if data['employment_type'] == 'Contract':
        res['employment_type_Contract'] = True
    elif data['employment_type'] == 'Freelance':
        res['employment_type_Freelance'] = True
    elif data['employment_type'] == 'Full-time':
        res['employment_type_Full-time'] = True
    else:
        res['employment_type_Part-time'] = True

    if data['work_setting'] == 'Hybrid':
        res['work_setting_Hybrid'] = True
    elif data['work_setting'] == 'In-person':
        res['work_setting_In-person'] = True
    else:
        res['work_setting_Remote'] = True

    # Set value for job title encoded feature
    job_title = data['job_title_encoded']
    job_title_mean_salary = df[df['job_title'] == job_title]['salary_in_usd'].mean()
    res['job_title_encoded'] = job_title_mean_salary
    print(res['job_title_encoded'])

    # Set value for job category encoded feature
    job_category = data['job_category_encoded']
    job_category_mean_salary = df[df['job_category'] == job_category]['salary_in_usd'].mean()
    res['job_category_encoded'] = job_category_mean_salary
    print(res['job_category_encoded'])

    # Set value for company location encoded feature
    company_location = data['company_location_encoded']
    company_location_mean_salary = df[df['company_location'] == company_location]['salary_in_usd'].mean()
    res['company_location_encoded'] = company_location_mean_salary
    print(res['company_location_encoded'])


    # Use the model to predict salary
    prediction = model.predict([list(res.values())])  # Wrap in a list

    # Convert prediction result to a serializable format
    prediction = prediction.tolist()[0]  # Convert to list and extract the value

    # Return prediction as JSON response
    return jsonify({'predicted_salary': prediction})

if __name__ == '__main__':
    app.run(debug=True)
