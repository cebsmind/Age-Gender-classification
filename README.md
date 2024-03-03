# Deep Learning Project : Age & Gender Classification
This Project's goal is to use Deep Learning to predict the age & gender of a person with a picture. Which is a difficult task even for a human (specially for age)

# App preview
![image](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/07b2aebc-130b-4594-b337-7058a6882655)

# Data Set Informaton

### [Data Link From Kaggle](https://www.kaggle.com/datasets/samuelagyemang/utkface)

- UTKFace dataset is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity. The images cover large variation in pose, facial expression, illumination, occlusion, resolution, etc. This dataset could be used on a variety of tasks, e.g., face detection, age estimation, age progression/regression, landmark localization, etc.
- More info here : https://susanqq.github.io/UTKFace/

# What is a CNN ?
![1 7_BCJFzekmPXmJQVRdDgwg](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/d1f4ffa5-33b4-421a-80fc-b7fa5368c40b)
A convolutional neural network (CNN) is a category of machine learning model, namely a type of deep learning algorithm well suited to analyzing visual data. CNNs, sometimes referred to as convnets, use principles from linear algebra, particularly convolution operations, to extract features and identify patterns within images. Although CNNs are predominantly used to process images, they can also be adapted to work with audio and other signal data.

# Getting Started
## First step : Configure GPU
Before delving into our deep learning project, it's essential to configure your GPU to ensure optimal performance. To check the availability of your GPU and prevent Out-Of-Memory (OOM) errors, you can run the following code snippet:
```python
import tensorflow as tf

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
    
# Check the list of available GPUs
tf.config.list_physical_devices('GPU')
# Output should resemble: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
More info about GPU configuration for tensorflow [Here](https://www.tensorflow.org/guide/gpu?hl=fr)

# Set up Data
## How to works with images ?
There are many ways, the one i adopted is to create a dataframe, in which we have the link of the URL as a column, the age, and the gender.
The labels of each face image is embedded in the file name, formated like `[age]_[gender]_[race]_[date&time].jpg`
- **[age]** is an integer from 0 to 116, indicating the age
- **[gender]** is either 0 (male) or 1 (female)
- **[race]** is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
- **[date&time]** is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace
In our case, we only going to use the **age** and **gender** information.

Here is how we can make this :
```python
# Directory containing the images
image_dir = 'UTKFace'

# Function to parse age and sex from file names
def parse_age_and_sex(filename):
    parts = filename.split('_')
    age = int(parts[0])
    sex = int(parts[1])
    return age, sex

# Create a DataFrame with file paths, ages, and sexes
file_paths = []
ages = []
sexes = []

for filename in os.listdir(image_dir):
    file_path = os.path.join(image_dir, filename)
    age, sex = parse_age_and_sex(filename)
    file_paths.append(file_path)
    ages.append(age)
    sexes.append(sex)

# Organize DataFrame columns and display the first few rows
df = pd.DataFrame({'path': file_paths, 'age': ages, 'sex': sexes})
df.head()
```
### Result : 
![image](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/cf5fedb1-a11c-41b9-af03-dbc3d7bc2b54)

We can also display some images now to see how they looks
### 25 yo male :
![image](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/bcc6e775-f556-4f3e-bad9-61f5c5446665)

### 14 yo female :
![image](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/cb2b0071-8ae5-46dd-9cd1-9c7ee21499a5)

# Pre-Processing
## 1. Train / Test
The first thing we need to do is to separate our datas into training set and test set, for age and gender. 
```python
# 'path' column contains file paths of images / 'Ages' and 'Genders' are target variables
x_image_paths = df['path']
y_age = df['age']
y_gender = df['sex']

# Split the dataset for age prediction
x_train_age, x_test_age, y_train_age, y_test_age = train_test_split(x_image_paths, y_age, test_size=0.2, random_state=42)

# Split the dataset for gender prediction
x_train_gender, x_test_gender, y_train_gender, y_test_gender = train_test_split(x_image_paths, y_gender, test_size=0.2, random_state=42)
```
## 2. Data Augmentation
In the context of our age and gender classification project using the UTKFaces dataset, data augmentation plays a crucial role in enhancing the robustness and generalization of our deep learning models. Data augmentation involves applying various transformations to the original images, creating augmented versions that maintain the same semantic content while introducing diversity. 
### Benefits of Data Augmentation
**Increased Dataset Size:**
- The UTKFaces dataset may be limited in size, and data augmentation allows us to artificially expand our dataset by generating new variations of the existing images.

**Improved Generalization:**
- Augmenting the data helps the model generalize better to unseen data by exposing it to a broader range of variations, such as different poses, lighting conditions, and facial expressions.

**Reduced Overfitting:**
- Data augmentation serves as a regularization technique, helping to prevent overfitting by exposing the model to a more diverse set of training examples.
  
**Invariant Learning:**
- Augmentation techniques, such as rotation, flipping, and scaling, make the model more invariant to these transformations during training, improving its ability to recognize patterns in various orientations.

### Example 
![image](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/9347fa2a-ae5e-4abc-9c78-f768c8dfa4af)

# Modelisation
## 1. Age prediction
My goal here was to use the pre-trained VGGFace model, designed for face recognition. But to be optimal with our task, I decided to fine tune during training it so it can adapt to our task for age & gender recognition.
### 1. VGG Face implementation
```python
# Load the VGGFace model with pre-trained weights
base_model = VGGFace(model='vgg16', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')
```
- **VGGFace** is a pre-trained model specifically designed for face recognition tasks.
- **model='vgg16'** specifies that the VGG16 architecture should be used.
- **include_top=False** excludes the final fully connected layers of the original VGG16 model.
- **input_shape=(IMG_SIZE, IMG_SIZE, 3)** defines the input shape for the model.
- **pooling='avg'** specifies global average pooling for reducing spatial dimensions.

### 2. Unfreeze top layers
```python
# Unfreeze the last few layers of the VGGFace model
for layer in base_model.layers[-3:]:
    layer.trainable = True
```
This code unfreezes the last three layers of the VGGFace model, allowing them to be fine-tuned during training.
### 3. Add layers
```python
# Create a new model for age prediction
model = Sequential()
model.add(base_model)  # VGGFace base model
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear')) #final dense layer for prediction (1 number)
```
- The pre-trained **VGGFace** model is added as the base layer.
- Several fully connected layers (**Dense**) are added on top of the VGGFace model for age prediction.
- **Dropout** layers are included for regularization to prevent overfitting.

### 4. Compile model
```python
# Define the learning rate
learning_rate = 0.001

# Create the Adam optimizer with the specified learning rate
optimizer = Adam(learning_rate=learning_rate)

# Compile the model
model.compile(optimizer=optimizer, loss='mean_squared_error')  # Use appropriate loss for regression
```
- The **Adam optimizer** is used with a specified learning rate.
- The model is compiled using mean squared error **('mean_squared_error')** as the loss function. This is common for regression tasks, such as age prediction.

### 5. Train model
```python
# Train the model
history_age = model.fit(age_train_generator, epochs=30, validation_data=age_test_generator)
```
In the training process of a deep learning model, an epoch refers to one complete pass through the entire training dataset. During each epoch, the model learns from the entire dataset, adjusting its weights based on the calculated loss and the optimization algorithm used

## 2. Gender classification
We do the same process, but instead we change the last layer when building the model :
```python
gender_model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification (0 or 1)
```

and we also change the `loss function` as we predict a binary class 
```python
# Compile the gender model
gender_model.compile(optimizer=optimizer_gender, loss='binary_crossentropy', metrics=['accuracy'])
```
# Evaluate model
## 1. Metrics
Now we trained our model, we can evaluate the metrics for each epochs for our age and gender models.
### 1. Age model loss function
![image](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/5da4f205-5bb2-4d4d-bac7-9ec5c115b17a)

We can see that the loss function is decreasing which is a good sign for our age prediction model.

### 2. Gender model accuracy 

![image](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/de2ef699-e348-450e-998e-3765db7cfdf6)

The gender model has already more than 90% of accuracy in the first epochs, but it seems that he can't get more because of overfitting. I kept this model as I don't have enough ressource to fine tune the model. But still a good start

## 2. Save models
It's essential to save models as it very long to train them
```python
#save model
model.save('age_model.h5')
gender_model.save('gender_model.h5')
```

## 3. Test model
To test our model, I defined a function `process_and_predict` 

```python
def process_and_predict(file, box_expansion=0.07, margin=1):
    im = Image.open(file)
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
        # If no face is detected, resize the entire image to 200x200
        im = im.resize((200, 200), resample=Image.BICUBIC)

    # Convert the PIL image to a NumPy array
    ar = np.asarray(im)

    # Convert the data type of the array to float32
    ar = ar.astype('float32')

    # Normalize the pixel values to the range [0, 1]
    ar /= 255.0

    # Reshape the array to match the expected input shape of the model
    ar = ar.reshape(-1, 200, 200, 3)

    # Predict age and gender using the provided models
    age = age_model.predict(ar)
    gender = np.round(gender_model.predict(ar))

    # Convert gender prediction to 'male' or 'female'
    gender = 'male' if gender == 0 else 'female'

    # Print the predicted age and gender
    print('Age:', int(age), '\nGender:', gender)

    # Return the resized image (300x300)
    return im.resize((300, 300), Image.BICUBIC)
```

The **process_and_predict** function takes an image file as input and performs the following steps:

**1.** Opens the image file and converts it to OpenCV format (BGR).

**2.** Utilizes the dlib face detector to identify faces in the image.

**3.** If a face is detected:
- Expands the bounding box around the face.
- Resizes the face to 200x200 pixels.

**4.** If no face is detected:
- Resizes the entire image to 200x200 pixels.

**5.** Converts the processed image to a NumPy array, normalizes pixel values, and reshapes it to match the expected input shape.

**6.** Uses pre-trained models for age and gender prediction:
- Predicts the age using the age_model.
- Predicts the gender and converts the prediction to 'male' or 'female' using the gender_model.

**7.** Prints the predicted age and gender.

**8.** Returns the resized image (300x300 pixels).

To use this function, we need to provide the path to our pre-trained age and gender models (age_model and gender_model), and call the function with the image file we want to analyze.

## 4. Test Samples
### 1. Baby 

```python
img_path = "test_images/1yo.jpg"  
process_and_predict(img_path)
```
it returns :

![image](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/d8f95b3a-ab17-44b9-8d40-45334c9f6024)


```
Age: 1 
Gender: female
```

### 2. 16 years old girl

![image](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/e6b41a76-434f-4103-98ae-609ea52568ac)

```python
Age: 23 
Gender: female
```

### 3. 20 years old girl

![image](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/be674faf-d80e-4e32-8678-4e6b6a8c9b86)

```python
Age: 21 
Gender: female
```

### 4. 25 years old male

![image](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/d5014a67-0ef3-4b1e-887f-9c602417f727)

```python
Age: 29 
Gender: male
```

### 5. 40 years old male
![image](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/294855a6-34b9-455f-8127-87f955e0b4ca)

```python
Age: 51 
Gender: male
```

### 6. 38 years old male

![image](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/37c84341-88a8-48ea-9d45-663516125da6)

```python
Age: 44 
Gender: male
```

### 7. 55 years old female

![image](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/4dac8836-1df8-4c3c-9c44-fce126514baa)

```python
Age: 75 
Gender: female
```

### 8. 70 years old female

![image](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/1863b0e8-6323-4767-804f-938c330de53e)

```python
Age: 81 
Gender: female
```

### 9. Male with make up

![image](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/de809e72-1301-4700-91df-661ea3add05e)

```python
Age: 21 
Gender: male
```

### Women with baby face

![image](https://github.com/cebsmind/Age-Gender-classification/assets/154905924/96186a7f-137a-486b-8218-3e735f4f27ca)

```python
Age: 8 
Gender: female
```

We can test with more images, but I just wanted to have a quick look on how it performs and it's not that bad as I didn't fine tune the model too much so it leaves us room for better performance.
The model perform pretty well for gender recognition, for age of course it's way harder even for a human to guess the right age but it's not incoherent at all. Except for some exception as we can see in the last picture where it predicts **8yo** but we, as human, see that's a grown woman. Overall, it's pretty satisfiying.

# Set up APP

## To run the app you need 
#### 2. Download both `age_model.h5` and `gender_model.h5`
#### 3. Set up folders : Below is the suggested folder structure for organizing your Flask app

```plaintext
flask-app/
│
├── models/
│   ├── age_model.h5
│   └── gender_model.h5
│
├── static/
│   ├── uploads/
│   │
│   ├── background/
│   │   ├── background.jpg
│   │ 
│   ├── css/
│   │   ├── style.css
│   │   
│   └── js/
│       └── script.js
│
├── templates/
│      ├── index.html
│      └── result.html
│
├── main.py
├── requirements.txt
├── .gitignore.txt
├── .gcloudignore
```
#### 4. Install dependencies 
- `python -m venv env`
- `.env/Scripts/activate`
- `pip install -r requirements.txt`
#### 5. Run in terminal 
- `python main.py`
- open http://127.0.0.1:5000/





