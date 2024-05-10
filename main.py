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
    enrolled_users = [1, 4, 7, 12, 19]
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            fingerprint_id = int(filename[:-4])  # Remove ".tif" extension and convert to integer
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            lbp_features = extract_lbp_features(image)
            fingerprints.append(lbp_features)

            if fingerprint_id / 10 == enrolled_users:
                labels.append(1)
            else:
                labels.append(0)

    return np.array(fingerprints), np.array(labels)


def extract_lbp_features(image):

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


def main():

    dataset_directory = r"C:\Users\ASUS\Desktop\Python tingz\FingerprintAuthenticationSystem\FVC_Dataset"
    features, labels = read_fingerprint_images(dataset_directory)
    X_train, X_test, y_train, y_test = split_data(features, labels)
    classifier = train_classifier(X_train, y_train)
    accuracy = evaluate_classifier(classifier, X_test, y_test)
    print("Accuracy:", accuracy)

    if __name__ == "__main__":
        main()


