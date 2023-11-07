import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np

# Load the dataset
df = pd.read_csv(r'C:\Users\krisd\OneDrive\Desktop\meta-album-master\Code\Data\BCT\labels.csv')
images_dir = r'C:\Users\krisd\OneDrive\Desktop\meta-album-master\Code\Data\BCT\images\\'

# Function to load and preprocess images
def load_images(image_files):
    images = []
    for file in image_files:
        # Open the image file, resize it to 128x128 and append to the list
        img = Image.open(images_dir + file).resize((128, 128))
        images.append(np.array(img))
    # Convert the list of images to a numpy array
    return np.array(images)

# Load all images as per the FILE_NAME column
X = load_images(df['FILE_NAME'].values)

# Encode category labels to integers
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['CATEGORY'])
# Convert integer encoded labels to one-hot encodings
y = to_categorical(integer_encoded)

# Get the number of classes from the shape of the one-hot encoded array
num_classes = y.shape[1]

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define data augmentation strategies for training data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
# Define data augmentation strategy for validation data (only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)

# Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # Output layer with 'softmax' for multi-class classification

# Compile the model with 'adam' optimizer and 'categorical_crossentropy' loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the training data generator and validate using the validation data generator
model.fit(train_datagen.flow(X_train, y_train, batch_size=32), validation_data=val_datagen.flow(X_val, y_val), epochs=10)

# Evaluate the model's performance on the validation set
scores = model.evaluate(val_datagen.flow(X_val, y_val))
print('Validation loss:', scores[0])  # Print the validation loss
print('Validation accuracy:', scores[1])  # Print the validation accuracy
