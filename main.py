import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern
from sklearn.metrics import roc_curve, accuracy_score


# Step 1: Read fingerprint images and extract LBP features
def read_fingerprint_images(directory):
#reads fingerprint images, returns the an aray of the extracted features of each image
# and an array indicating whether they are a geniune/enrolled user; label[i] = 1, or an imposter; label[i] = 0
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
#extract features using |Texture-based extraction|
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
#splits data into Train:Test in a 60:40 split
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

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Calculate False Match Rate (FMR) and False Non-Match Rate (FNMR)
    genuine_indices = (y_test == 1)
    imposter_indices = (y_test == 0)
    genuine_scores = classifier.predict_proba(X_test[genuine_indices])[:, 1]
    imposter_scores = classifier.predict_proba(X_test[imposter_indices])[:, 1]

    fmr, fnmr = calculate_fmr_fnmr(genuine_scores, imposter_scores)

    # Plot ROC curve
    plot_roc_curve(genuine_scores, imposter_scores)

    # Determine Equal Error Rate (EER)
    eer = calculate_eer(fmr, fnmr)

    return accuracy, fmr, fnmr, eer


def calculate_fmr_fnmr(genuine_scores, imposter_scores):
    thresholds = np.linspace(0, 1, 100)
    fmr = []
    fnmr = []
    for threshold in thresholds:
        fmr.append(np.mean(imposter_scores >= threshold))
        fnmr.append(np.mean(genuine_scores < threshold))
    return np.array(fmr), np.array(fnmr)


def calculate_eer(fmr, fnmr):
    eer_threshold = np.argmin(np.abs(fmr - fnmr))
    eer = (fmr[eer_threshold] + fnmr[eer_threshold]) / 2
    return eer


def plot_roc_curve(genuine_scores, imposter_scores):
    fpr, tpr, thresholds = roc_curve(np.concatenate([np.ones_like(genuine_scores), np.zeros_like(imposter_scores)]),
                                     np.concatenate([genuine_scores, imposter_scores]))
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.show()


def main():
    dataset_directory = r"C:\Users\ASUS\Desktop\Python tingz\FingerprintAuthenticationSystem\FVC_Dataset"
    features, labels = read_fingerprint_images(dataset_directory)
    X_train, X_test, y_train, y_test = split_data(features, labels)
    classifier = train_classifier(X_train, y_train)
    accuracy, fmr, fnmr, eer = evaluate_classifier(classifier, X_test, y_test)

    print("Accuracy:", accuracy)
    print("FMR:", fmr)
    print("FNMR:", fnmr)
    print("EER:", eer)



if __name__ == "__main__":
    main()
