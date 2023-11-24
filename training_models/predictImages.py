from keras.models import load_model
from keras.preprocessing import image as image_utils
import PIL.ImageOps
from tools import tools as t
from tools.ImageDescriptor import describe
import numpy as np
import joblib

def predict_image_with_CNN(path, model):
    img = image_utils.load_img(path, target_size=(100, 100))
    img = PIL.ImageOps.invert(img)
    img = image_utils.img_to_array(img)
    img = img / 255.0
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

    predictions = model.predict(img, verbose=0, batch_size=1)
    predicted_class = np.argmax(predictions, axis=1)[0]

    return path, t.classes[predicted_class]

def predict_image_with_RF(org_path, cropped_path, clf):
    feature = describe(org_path, cropped_path)
    y = clf.predict(feature.reshape(1, -1))
    return org_path, t.classes[y[0]]

def main():
    cnn_model_path = "/models/modelCNN/size100/trainedModelE40.h5"
    rf_model_path = "/models/modelRF/rf_model_final.joblib"
    image_path = "5176841.jpg"  # Change the image path

    cnn_model = load_model(cnn_model_path)
    rf_model = joblib.load(rf_model_path)

    cnn_prediction = predict_image_with_CNN(image_path, cnn_model)
    print("CNN Prediction:", cnn_prediction)

    cropped_path = '/train_data/img_for_train/cropped'
    rf_prediction = predict_image_with_RF(image_path, cropped_path, rf_model)
    print("RandomForest Prediction:", rf_prediction)

if __name__ == "__main__":
    main()
