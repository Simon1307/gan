import os
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, render_template, send_file, send_from_directory
from torchvision import transforms
from src.components.model import Generator, ConvBlock, ResidualBlock
from src.logger import logging
from src.utils import Config, load_model
from datetime import datetime


app = Flask(__name__)

# Load the pre-trained PyTorch model
model = load_model(model_path='artifacts/gen_M.pth', device='cpu')
model.eval()
logging.info('Pretrained model loaded')

config = Config()
pred_transformations = config.pred_transforms

# Define the image preprocessing function
def preprocess_image(input_image):
    transform = pred_transformations
    input_tensor = transform(input_image).unsqueeze(0)  # Add batch dimension
    return input_tensor

# Define a route for the web app home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle image processing
@app.route('/process', methods=['POST'])
def process():
    # Get the uploaded image from the request
    uploaded_image = request.files['input_image']
    if uploaded_image:
        input_image = Image.open(uploaded_image)
        input_tensor = preprocess_image(input_image)

        logging.info('Passing image through model')
        # Pass the input through the model
        with torch.no_grad():
            output_tensor = model(input_tensor)

        # Denormalize the output tensor
        output_tensor = (output_tensor + 1) / 2.0

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # Save the input image with a unique identifier
        input_filename = f'input_{timestamp}.jpg'
        input_image_path = os.path.join('static', 'input', input_filename)
        input_image.save(input_image_path)

        # Save the generated image with a unique identifier
        output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
        output_filename = f'output_{timestamp}.jpg'
        output_image_path = os.path.join('static', 'output', output_filename)
        output_image.save(output_image_path)
        logging.info('Input image and generated image were saved')

    return render_template('index.html', input_filename=input_filename, output_filename=output_filename)


# Routine for downloading the generated image
@app.route('/download/<filename>')
def download(filename):
    return send_from_directory('static/output', filename, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
