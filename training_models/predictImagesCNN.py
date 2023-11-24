import cv2
from keras.models import load_model
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
import PIL.ImageOps
import tools as T
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib


def predict_image_with_CNN(path, model):
    # Load the image using ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1. / 255)
    img = image_utils.load_img(path, target_size=(100, 100))
    img = PIL.ImageOps.invert(img)

    # Convert the image to a numpy array
    img_array = image_utils.img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)

    # Use the ImageDataGenerator to preprocess the image
    img_array = datagen.standardize(img_array)

    # Make predictions
    predictions = model.predict(img_array, verbose=0, batch_size=1)
    predicted_class = np.argmax(predictions, axis=1)[0]

    return path, T.classes[predicted_class]


def main():
    # Replace these paths with the actual paths to your all_train_data and images
    cnn_model_path = "modelCNN/size100/trainedModelE20.h5"
    rf_model_path = "rf_model_final.joblib"
    image_path = "4658624.jpg"

    # Load CNN train
    cnn_model = load_model(cnn_model_path)

    # Predict with CNN
    cnn_prediction = predict_image_with_CNN(image_path, cnn_model)
    print("CNN Prediction:", cnn_prediction)



if __name__ == "__main__":
    main()
