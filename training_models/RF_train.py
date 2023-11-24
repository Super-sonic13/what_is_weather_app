import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import numpy as np
import tools as T

def train_rf_on_batches(data, data_label, batch_size=100, n_estimators=12):
    # K-fold splits into 10 and shuffles the indexes
    split_size = 10
    kf = KFold(n_splits=split_size, shuffle=True)
    kf.get_n_splits(data)

    overallAccuracies = np.zeros(5)
    generalOverallAccuracy = 0

    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        train_label, test_label = data_label[train_index], data_label[test_index]

        # Create a random forest Classifier. By convention, clf means 'Classifier'
        clf = RandomForestClassifier(
            bootstrap=False,
            max_leaf_nodes=None,
            n_estimators=n_estimators,
            min_weight_fraction_leaf=0.0,
        )

        # Process the data in smaller batches
        for batch_start in range(0, len(train_data), batch_size):
            batch_end = batch_start + batch_size
            batch_train_data = train_data[batch_start:batch_end]
            batch_train_label = train_label[batch_start:batch_end]

            # Reshape batch features
            batch_train_data_reshaped = batch_train_data.reshape((len(batch_train_data), -1))

            # Train the Classifier on the batch
            if len(batch_train_data_reshaped) > 0:
                clf.fit(batch_train_data_reshaped, batch_train_label)

        # Evaluate the train on the test set
        test_data_reshaped = test_data.reshape((len(test_data), -1))
        generalAccuracy = T.get_accuracy_of_class(test_label, clf.predict(test_data_reshaped))
        generalOverallAccuracy += generalAccuracy
        print("General Accuracy:", generalAccuracy)
        print("---------------------------------------------")

    print("OVERALL ACCURACY\n-------------------------------------------------")
    for i in range(len(overallAccuracies)):
        print("Overall Accuracy For", T.classes[i], overallAccuracies[i] / split_size)
    print("Overall Accuracy", generalOverallAccuracy / split_size)

    # Save the final trained train
    joblib.dump(clf, 'rf_model_final.joblib')

# Example usage
data_file = f'models/all_train_data.npy'
label_file = f'models/all_train_label.npy'

data = np.load(data_file)[:2000]
data_label = np.load(label_file)[:2000]

data_label = np.reshape(data_label, (np.shape(data)[0],))

# Train the train using smaller batches
train_rf_on_batches(data, data_label, batch_size=100, n_estimators=12)
