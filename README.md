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
