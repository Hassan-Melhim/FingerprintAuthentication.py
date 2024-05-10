import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern


# Step 1: Read fingerprint images and extract LBP features
def read_fingerprint_images(directory):
    fingerprints = []
    labels = []
    fingerprint_counts = {}  # Dictionary to store the count of samples for each fingerprint ID
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            fingerprint_id = int(filename[:-4])  # Remove ".tif" extension and convert to integer
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            lbp_features = extract_lbp_features(image)
            fingerprints.append(lbp_features)
            labels.append(fingerprint_id)
            # Increment the count for the current fingerprint ID
            fingerprint_counts[fingerprint_id] = fingerprint_counts.get(fingerprint_id, 0) + 1

    # Filter out fingerprint IDs with fewer than 8 samples
    filtered_features = []
    filtered_labels = []
    for feature, label in zip(fingerprints, labels):
        if fingerprint_counts[label] == 8:
            filtered_features.append(feature)
            filtered_labels.append(label)

    return np.array(filtered_features), np.array(filtered_labels)


def extract_lbp_features(image):
    # Convert the image to grayscale if it's not already
    if image.mode != 'L':
        image = image.convert('L')

    # Convert PIL image to numpy array
    image_array = np.array(image)

    # Extract LBP features
    radius = 3
    n_points = 8 * radius
    lbp_image = local_binary_pattern(image_array, n_points, radius, method='uniform')

    # Flatten the LBP image to get feature vector
    lbp_features = lbp_image.ravel()

    return lbp_features


# Step 2: Split data into training and testing sets
def split_data(features, labels):
    # Ensure that there are at least 2 fingerprint IDs with 8 samples each
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if np.sum(label_counts == 8) < 2:
        raise ValueError(
            "Insufficient samples for splitting. Ensure that there are at least 2 fingerprint IDs with 8 samples each.")

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42,
                                                        stratify=labels)
    return X_train, X_test, y_train, y_test


# Step 3: Train a classifier
def train_classifier(X_train, y_train):
    classifier = SVC(kernel='linear')
    classifier.fit(X_train, y_train)
    return classifier


# Step 4: Evaluate the classifier
def evaluate_classifier(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


#Example usage
dataset_directory = r"C:\Users\ASUS\Desktop\Python tingz\FingerprintAuthenticationSystem\FVC_Dataset"
features, labels = read_fingerprint_images(dataset_directory)
X_train, X_test, y_train, y_test = split_data(features, labels)
classifier = train_classifier(X_train, y_train)
accuracy = evaluate_classifier(classifier, X_test, y_test)
print("Accuracy:", accuracy)
