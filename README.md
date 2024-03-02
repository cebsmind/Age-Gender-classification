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
