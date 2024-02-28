from flask import Flask, render_template, request, url_for  
import os
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import cv2
import dlib
from keras.models import load_model

upload_folder = os.path.join('static', 'uploads')

app = Flask(__name__)

# Define the upload folder
app.config['UPLOAD_FOLDER'] = upload_folder

# Obtenez le chemin absolu du dossier des modèles
models_dir = os.path.join(os.path.dirname(__file__), 'models')

# Utilisez des chemins relatifs pour les modèles
age_model_path = os.path.join(models_dir, 'age_model.h5')
gender_model_path = os.path.join(models_dir, 'gender_model.h5')

# Importer le modèle age 
age_model = load_model(age_model_path)
# Importer le modèle gender
gender_model = load_model(gender_model_path)

def process(file, box_expansion=0.05, margin=5):
    # Check if the uploaded file is an image
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
        # Open the image file using PIL
        try:
            im = Image.open(file)
        except Exception as e:
            print("Error opening image:", e)
            return None

        # Convert PIL image to OpenCV format (BGR)
        cv_image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

        # Create the Dlib face detector
        detector = dlib.get_frontal_face_detector()

        # Detect faces in the image
        faces = detector(cv_image)

        if faces:
            # Use the first detected face for simplicity
            face_rect = faces[0]

            # Convert Dlib rectangle to (x, y, w, h)
            x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()

            # Expand the bounding box
            expansion_x = int(w * box_expansion)
            expansion_y = int(h * box_expansion)
            x -= expansion_x
            y -= expansion_y
            w += 2 * expansion_x
            h += 2 * expansion_y

            # Add margin
            x -= margin
            y -= margin
            w += 2 * margin
            h += 2 * margin

            # Ensure the expanded box is within the image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(cv_image.shape[1] - x, w)
            h = min(cv_image.shape[0] - y, h)

            # Crop and zoom on the expanded face
            face = cv_image[y:y+h, x:x+w]

            # Resize the face to 200x200
            face = cv2.resize(face, (200, 200))

            # Convert the NumPy array back to PIL image
            im = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        else:
            # If no face is detected, display an error message and return None
            print("No face was detected in the image.")
            return None

        # Convert the PIL image to a NumPy array
        ar = np.asarray(im)

        # Check if the conversion resulted in a valid array
        if ar is None:
            print("Error converting image to array.")
            return None

        # Convert the data type of the array to float32
        ar = ar.astype('float32')

        # Normalize the pixel values to the range [0, 1]
        ar /= 255.0

        # Reshape the array to match the expected input shape of the model
        ar = ar.reshape(-1, 200, 200, 3)

        return ar

    else:
        print("Invalid file extension.")
        return None

# Define a route for handling both GET and POST requests to the root URL
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Handle the POST request here (i.e., when the form is submitted)
        file = request.files['file']
        if file:
            # Process and predict with the file 
            ar = process(file)
            age = age_model.predict(ar)
            gender = np.round(gender_model.predict(ar))
            if gender == 0:
                gender = 'Male'
            elif gender == 1:
                gender = 'Female'
            return render_template('result.html', age=age, gender=gender, image_path=url_for('static', filename=f'uploads/{file.filename}'))
    
    # Handle the GET request here (i.e., when the page is initially loaded)
    return render_template('index.html')

@app.route('/detect_age_gender', methods=['GET', 'POST'])
def detect_age_gender():
    if request.method == 'POST':
        file = request.files['file']

        # Save the file to the 'static/uploads' folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)

        # Use process function
        ar = process(file)

        if ar is None:
            return "Erreur lors du traitement de l'image : veuillez vérifier que la photo est dans un format valide et qu'un visage est présent."

        # Predict age and gender using the provided models
        age = age_model.predict(ar)
        gender_prediction = np.round(gender_model.predict(ar))

        if gender_prediction not in [0, 1]:
            return "Erreur lors de la prédiction du genre : le modèle a renvoyé une valeur inattendue."

        # Convert gender prediction to 'Male' or 'Female'
        gender = 'Male' if gender_prediction == 0 else 'Female'

        # Return result to the "result" page HTML
        return render_template('result.html', age=age, gender=gender, image_path=url_for('static', filename=f'uploads/{file.filename}'))
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)