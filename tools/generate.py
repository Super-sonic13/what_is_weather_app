import tools as T

# Define your classes and corresponding folders
classes_and_folders = {
    "Cloudy": "0",
    "Sunny": "1",
    "Rainy": "2",
    "Snowy": "3",
    "Foggy": "4"
}
'''
# Set your image root, cropped destination, and size
image_root = "train/"
cropped_dest = "cropped/"
size = 100

# Iterate through classes and prepare the dataset
for class_name, folder_name in classes_and_folders.items():
    class_image_root = image_root + folder_name + "/"
    class_cropped_dest = cropped_dest + folder_name + "/"

    T.prepare_data_set(class_image_root, class_cropped_dest, size)



# Set your image root, destination, and size for image_to_matrix
image_root = "cropped/"
model_dest = "train/"
size = 100

# Create .npy files for image_to_matrix
T.image_to_matrix(image_root=image_root, dest=model_dest, size=size)'''
# Concatenate the datasets
T.concatenate()