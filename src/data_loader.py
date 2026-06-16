#Iris loading + preprocessing


from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def normalize(X):
    # Standardization (Z-score normalization)
    # μ = mean of feature
    # σ = standard deviation of feature
    # x' = (x - μ) / σ

    # X shape: (N, D) -> (150, 4)
    mu  = np.mean(X, axis=0)  # mean of each feature (4,)
    sigma = np.std(X, axis=0)  # std of each feature (4,)
    return (X - mu) / sigma  # normalized features (150, 4)

def get_data():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data  # Features (150, 4)
    y = iris.target  # Labels (150,)

    print(y.shape)

    # print(f"Original features shape: {X.shape}")
    # print(f"Original labels shape: {y.shape}")
    print(f"Original features sample:\n{X[:2]}")
    print(f"Original labels sample:\n{y[:2]}")

    # # normalize input features   
    X = normalize(X)

    #print(f"Normalized features shape: {X.shape}")
    #print(f"Normalized features sample:\n{X[:10]}")

    #train test split   
    X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return (
        np.array(X_train),
        np.array(X_test),
        np.array(y_train),
        np.array(y_test)
    )

#get_data()