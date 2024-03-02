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
The first thing we need to do is to separate our datas into train/test for age & gender, like this : 
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


