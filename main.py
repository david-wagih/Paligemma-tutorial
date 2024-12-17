import os
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
import keras
import keras_hub
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure Keras
keras.config.set_floatx("bfloat16")

# Initialize PaliGemma model
def init_model():
    try:
        return keras_hub.models.PaliGemmaCausalLM.from_preset("pali_gemma_3b_mix_224")
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        raise

# Utility functions
def crop_and_resize(image, target_size):
    width, height = image.size
    source_size = min(image.size)
    left = width // 2 - source_size // 2
    top = height // 2 - source_size // 2
    right, bottom = left + source_size, top + source_size
    return image.resize(target_size, box=(left, top, right, bottom))

def process_image(image_file, target_size=(224, 224)):
    # Open and process image
    image = Image.open(image_file)
    image = crop_and_resize(image, target_size)
    image = np.array(image)
    # Remove alpha channel if necessary
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        # Process image
        processed_image = process_image(image_file)
        
        # Get analysis type from form
        analysis_type = request.form.get('analysis_type', 'describe')
        
        # Initialize model (in practice, you'd want to cache this)
        model = init_model()
        
        # Prepare prompt based on analysis type
        if analysis_type == 'describe':
            prompt = 'describe en\n'
        elif analysis_type == 'detect':
            prompt = 'detect object\n'
        elif analysis_type == 'answer':
            question = request.form.get('question', 'What is in this image?')
            prompt = f'answer en {question}\n'
        
        # Generate output
        output = model.generate(
            inputs={
                "images": processed_image,
                "prompts": prompt,
            }
        )
        
        return jsonify({
            'result': output,
            'analysis_type': analysis_type
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Check for environment variables from .env
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')
    
    if not kaggle_username or not kaggle_key:
        print("Error: KAGGLE_USERNAME and KAGGLE_KEY must be set in .env file")
        print("Please create a .env file with these variables")
        exit(1)
    
    # Set Kaggle credentials from .env
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key
    
    # Create upload directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Get port from environment variable or use default
    port = int(os.getenv('PORT', 5000))
    
    app.run(debug=True, port=port)
