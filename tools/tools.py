import os
import random

import PIL.ImageOps
import cv2
import numpy as np
from PIL import Image
from keras.preprocessing import image as image_utils
from keras.utils import to_categorical

classes = ["Cloudy", "Sunny", "Rainy", "Snowy", "Foggy"]


def binary_to_class(label):
    """ Converts a binary class matrix to class vector(integer)
        # Arguments:
            label: matrix to be converted to class vector
    """
    new_lbl = []
    for i in range(len(label)):
        new_lbl.append(np.argmax(label[i]))
    return new_lbl


def get_accuracy_of_class(v_label, y):
    """
        Returns:
            accuracy of given label
        Args:
            validation label: expected outputs
            y: predicted outputs
    """
    c = 0
    for i in range(len(y)):
        if y[i] == v_label[i]:
            c += 1
    return c / len(y)


def separate_data(v_data, v_label):
    """separates validation data and label according to class no
        Args:
            v_data: validation data to be split
            v_label: validation label to be split
        Returns:
            an array that stores '[val_data,val_label]' in each index for each class.
    """
    vd = [[[], []] for _ in range(5)]
    for i in range(len(v_data)):
        cls = int(v_label[i])
        vd[cls][0].append(v_data[i])
        vd[cls][1].append(cls)
    for i in range(5):
        vd[i][0] = np.array(vd[i][0])
        vd[i][1] = np.array(vd[i][1])
    return vd


def __find_sky_area(path_of_image):
    read_image = cv2.imread(path_of_image, 50)
    edges = cv2.Canny(read_image, 150, 300)

    shape = np.shape(edges)
    left = np.sum(edges[0:shape[0] // 2, 0:shape[1] // 2])
    right = np.sum(edges[0:shape[0] // 2, shape[1] // 2:])

    if right > left:
        return 0  # if right side of image includes more building etc. return 0 to define left side(0 side) is sky area
    else:
        return 1  # if left side of image includes more building etc. return 1 to define right side(1 side) is sky area


import os

def resize_image(base_size, path_of_image, destination, new_image_name):
    img = Image.open(path_of_image)

    if img.size[0] >= img.size[1]:
        sky_side = __find_sky_area(path_of_image)
        base_height = base_size
        wpercent = (base_height / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(wpercent)))
        img = img.resize((wsize, base_height), Image.LANCZOS)
        if sky_side == 0:
            img = img.crop((0, 0, base_size, img.size[1]))
        else:
            img = img.crop((img.size[0] - base_size, 0, img.size[0], img.size[1]))

    else:
        base_width = base_size
        wpercent = (base_width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((base_width, hsize), Image.LANCZOS)
        img = img.crop((0, 0, img.size[0], base_size))

    # Ensure the destination directory exists
    os.makedirs(destination, exist_ok=True)

    img.save(os.path.join(destination, new_image_name))



def prepare_data_set(path, dest, size):
    # root directory for source images(which will be cropped)
    # path = '../train/1/'
    # root directory as destination to save cropped images(Prepared images will be saved in here)
    # dest = '../cropped100/1'

    for filename in os.listdir(path):
        resize_image(size,  # crop size for all images (just change it to define crop size)
                     path + filename,
                     dest,
                     filename)



from keras.preprocessing import image as image_utils

def image_to_matrix(image_root='cropped/', dest='train/', size=100, batch_size_for_models=5000):
    """
        reads all images in a directory given,
        adds it to an array and labels each image, then saves those train.
    """

    #image_root = "../cropped100/"  # Change this root directory of images to create train for them
    batch_size_for_models = 5000  # 5000 sized batch all_train_data

    train_data = []
    train_label = []

    # list of directory of classes in given path
    classes_dir = os.listdir(image_root)

    counter = 0  # counter to check size of batch, if 5000 save train and flush lists
    fc = 0  # file counter to name all_train_data
    for cls in classes_dir:
        class_list = os.listdir(image_root + cls + "/")  # image list in a class directory
        for imageName in class_list:
            counter += 1

            img = image_utils.load_img(image_root + cls + "/" + imageName, target_size=(size, size))  # open an image
            img = PIL.ImageOps.invert(img)  # inverts it
            img = image_utils.img_to_array(img)  # converts it to array

            train_data.append(img)
            train_label.append(int(cls))

            if counter == batch_size_for_models:
                train_data, train_label = shuffle(train_data, train_label)
                np.save(dest+"train_data" + str(fc) + ".npy",
                        np.array(train_data))  # train root to save image all_train_data(image)
                np.save(dest+"train_label" + str(fc) + ".npy",
                        np.array(train_label))  # train root to save image all_train_data(label))

                train_data = []
                train_label = []
                fc += 1
                counter = 0

    # rest of images which stays in list , add their all_train_data to train root lastly
    if len(train_data) != 0:
        train_data, train_label = shuffle(train_data, train_label)
        np.save(dest+"train_data.npy", np.array(train_data))  # train root to save image all_train_data(image)
        np.save(dest+"train_label.npy", np.array(train_label))  # train root to save image all_train_data(label)








def shuffle(data, label):
    temp = list(zip(data, label))
    random.shuffle(temp)
    return zip(*temp)


def concatenate():
    """
        concatenates the first 1000 images which separated for each class
    """
    train_data = np.load("train/train_data7.npy")[:1000]
    train_label = np.load("train/train_label7.npy")[:1000]

    temp_data = np.load("train/train_data8.npy")[:1000]
    temp_label = np.load("train/train_label8.npy")[:1000]

    train_data = np.concatenate((train_data, temp_data), axis=0)
    train_label = np.concatenate((train_label, temp_label), axis=0)

    temp_data = np.load("train/train_data.npy")
    temp_label = np.load("train/train_label.npy")

    train_data = np.concatenate((train_data, temp_data), axis=0)
    train_label = np.concatenate((train_label, temp_label), axis=0)

    train_data, train_label = shuffle(train_data, train_label)
    np.save("all_train_data/train_data_concat78.npy", train_data)
    np.save("all_train_data/train_label_concat78.npy", train_label)


import joblib

def load_rf_model(model_path):
    """
    Load a RandomForest train from the given file path.
    Args:
        model_path (str): The file path to the RandomForest train.
    Returns:
        RandomForest train: The loaded RandomForest train.
    """
    return joblib.load(model_path)

import numpy as np
from sklearn.utils import shuffle

def concatenate_all():
    """
    Concatenates all train data and label files.
    """
    all_train_data = []
    all_train_label = []

    for i in range(9):
        data_file = f'train/train_data{i}.npy'
        label_file = f'train/train_label{i}.npy'

        train_data_i = np.load(data_file)
        train_label_i = np.load(label_file)

        all_train_data.append(train_data_i)
        all_train_label.append(train_label_i)

    all_train_data = np.concatenate(all_train_data, axis=0)
    all_train_label = np.concatenate(all_train_label, axis=0)

    # Shuffle the concatenated data and labels
    all_train_data, all_train_label = shuffle(all_train_data, all_train_label)

    # Save the concatenated data and labels
    np.save("all_train_data/all_train_data.npy", all_train_data)
    np.save("all_train_data/all_train_label.npy", all_train_label)

# Викликати функцію для об'єднання всіх файлів

