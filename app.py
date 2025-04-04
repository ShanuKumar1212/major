# Flask application to support Treatment and Medicine Identification

from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# Load Excel data
EXCEL_FILE = "disease_treatment_dataset.xlsx"
data = pd.read_excel(EXCEL_FILE)

# Load trained model
MODEL_PATH = "trademed_mobilenet.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define Class Indices
class_indices = {
    "Arive-Dantu": 0,
    "Basale": 1,
    "Betel": 2,
    "Crape_Jasmine": 3,
    "Curry": 4,
    "Drumstick": 5,
    "Fenugreek": 6,
    "Guava": 7,
    "Hibiscus": 8,
    "Indian_Beech": 9,
    "Indian_Mustard": 10,
    "Jackfruit": 11,
    "Jamaica_Cherry-Gasagase": 12,
    "Jamun": 13,
    "Jasmine": 14,
    "Karanda": 15,
    "Lemon": 16,
    "Mango": 17,
    "Mexican_Mint": 18,
    "Mint": 19,
    "Neem": 20,
    "Oleander": 21,
    "Parijata": 22,
    "Peepal": 23,
    "Pomegranate": 24,
    "Rasna": 25,
    "Rose_apple": 26,
    "Roxburgh_fig": 27,
    "Sandalwood": 28,
    "Tulsi": 29
}
idx_to_class = {v: k for k, v in class_indices.items()}

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the uploaded image for prediction.
    Args:
        image: PIL Image object
        target_size: Tuple specifying target size for the model
    Returns:
        Numpy array suitable for model prediction
    """
    image = image.resize(target_size)  # Resize to the target size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/treatment', methods=['GET', 'POST'])
def treatment():
    if request.method == 'POST':
        try:
            disease = request.form.get('disease').strip()
            age = int(request.form.get('age'))
            gender = request.form.get('gender').strip().lower()
            level = request.form.get('disease_level').strip().lower()

            # Map disease level hierarchy
            level_hierarchy = {'low': 1, 'normal': 2, 'high': 3}

            # Filter data based on the conditions
            def match_row(row):
                # Check disease match
                if row['Disease'].strip().lower() != disease.lower():
                    return False
                
                # Check age within range
                try:
                    age_range = row['Age'].split('-')
                    min_age, max_age = int(age_range[0]), int(age_range[1])
                    if not (min_age <= age <= max_age):
                        return False
                except ValueError:
                    return False  # Skip if age format is invalid
                
                # Check gender match
                if row['Gender'].strip().lower() not in ['any', gender]:
                    return False
                
                # Check level of disease hierarchy
                if level_hierarchy[row['Level of Disease'].strip().lower()] > level_hierarchy[level]:
                    return False

                return True

            # Apply filtering to data
            filtered_data = data[data.apply(match_row, axis=1)]

            # Get treatment if a match is found
            if not filtered_data.empty:
                treatment_info = filtered_data.iloc[0]['Treatment']
                return jsonify({'treatment': treatment_info})

            # No matching treatment found
            return jsonify({'treatment': 'No Treatment Found'})
        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('treatment.html')

@app.route('/medicine-identification', methods=['GET', 'POST'])
def medicine_identification():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'image' in request.files:
            try:
                uploaded_image = request.files['image']
                image = Image.open(uploaded_image)

                # Preprocess the image and predict using the model
                preprocessed_image = preprocess_image(image)
                predictions = model.predict(preprocessed_image)
                class_idx = np.argmax(predictions)
                class_label = idx_to_class[class_idx]
                confidence = np.max(predictions) * 100

                return jsonify({
                    'result': class_label,
                    'confidence': f"{confidence:.2f}%"
                })
            except Exception as e:
                return jsonify({'error': str(e)})

        # If a text query is provided, handle it as before
        query = request.form.get('query')
        if query:
            medicines = data[data['Medicine'].str.contains(query, case=False)]
            response = [{'name': med, 'image': "placeholder.jpg"} for med in medicines['Medicine']]
            return jsonify(response)
    return render_template('medicine_identification.html')

@app.route('/disease-autocomplete', methods=['GET'])
def disease_autocomplete():
    diseases = data['Disease'].drop_duplicates().str.upper().tolist()
    return jsonify(diseases)

@app.route('/medicine-autocomplete', methods=['GET'])
def medicine_autocomplete():
    try:
        json_file_path = os.path.join(app.root_path, 'medicine_data.json')
        with open(json_file_path, 'r') as file:
            medicines = json.load(file)

        term = request.args.get('term', '').lower()
        filtered_medicines = [
            med for med in medicines if term in med['name'].lower()
        ]

        return jsonify(filtered_medicines)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
