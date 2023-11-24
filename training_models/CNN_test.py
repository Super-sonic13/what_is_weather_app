import numpy as np
from keras.utils import to_categorical

from keras.models import load_model
import tools as T


data_file = f'all_train_data/all_train_data.npy'
label_file = f'all_train_data/all_train_label.npy'
data = np.load(data_file)
data_label = np.load(label_file)
# normalization
data = data / 255.0


# each index stores a list which stores validation data and its label according to index no
# vd[0] = [val,lab] for class 0
# vd[1] = [val,lab] for class 1 and so on
vd = T.separate_data(data, data_label)

# number of class
num_classes = 5  # Cloudy,Sunny,Rainy,Snowy,Foggy

# for example if label is 4 converts it [0,0,0,0,1]
data = to_categorical(data, num_classes)



# loads trained train and architecture
model = load_model("modelCNN/size100/trainedModelE20.h5")


# -------predicting part-------
y = model.predict_classes(data_file, verbose=0)
acc = T.get_accuracy_of_class(T.binary_to_class(data), y)
print("General Accuracy for Validation Data:", acc)
print("-----------------------------")

for i in range(len(vd)):
    v_data = vd[i][0]
    v_label = vd[i][1]
    y = model.predict_classes(v_data, verbose=0)
    acc = T.get_accuracy_of_class(v_label, y)
    print("Accuracy for class " + T.classes[i] + ": ", acc)
    print("-----------------------------")