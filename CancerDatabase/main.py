import numpy as np
import pandas as pd
import time
from numpy import linalg
from cvxopt import matrix, solvers
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


# Step 1: Load and preprocess the dataset
def load_and_preprocess_data():
    # Load the breast cancer dataset from sklearn
    data = load_breast_cancer()
    
    # Create DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # Split into features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data into training and test sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Step 2: Generate Granular Balls (using a simplified approach)
def generate_granular_balls(X, y, purity_threshold=0.95):
    granular_balls = []
    
    # Combine data and labels for ball generation
    data = np.hstack((X, y.values.reshape(-1, 1)))
    
    # Apply KMeans clustering to the dataset (here, we assume two classes)
    kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
    
    # Create granular balls based on KMeans clustering
    for label in np.unique(kmeans.labels_):
        ball_data = data[kmeans.labels_ == label]
        purity = np.mean(ball_data[:, -1] == label)
        
        # Only keep the balls with purity above the threshold
        if purity >= purity_threshold:
            granular_balls.append(ball_data)
    
    return granular_balls

# Step 3: Hidden Matrix Generation using RVFL (Random Vector Functional Link)
def generate_hidden_matrix(X, N=23, activation_function=1):
    # Random initialization of weights and bias
    np.random.seed(0)
    W = np.random.rand(X.shape[1], N) * 2 - 1
    b = np.random.rand(1, N)
    
    # Apply activation function to the hidden layer
    X1 = np.dot(X, W) + np.tile(b, (X.shape[0], 1))
    
    if activation_function == 1:  # SELU
        X1 = np.maximum(0, X1) + np.minimum(0, 1.673 * (np.exp(X1) - 1))
    elif activation_function == 2:  # ReLU
        X1 = np.maximum(0, X1)
    elif activation_function == 3:  # Sigmoid
        X1 = 1 / (1 + np.exp(-X1))
    
    # Concatenate original data with transformed hidden matrix
    X_transformed = np.hstack((X, X1))
    
    return X_transformed

# Step 4: TWSVM Implementation
def TWSVM_main(DataTrain, TestX, d1, d2):
    mew = 1
    eps1 = 0.05
    eps2 = 0.05
    A = DataTrain[DataTrain[:, -1] == 1, :-1]
    B = DataTrain[DataTrain[:, -1] != 1, :-1]

    m1 = A.shape[0]
    m2 = B.shape[0]
    e1 = np.ones((m1, 1))
    e2 = np.ones((m2, 1))

    H1 = np.hstack((A, e1))
    G1 = np.hstack((B, e2))
    
    HH1 = np.dot(H1.T, H1) + eps1 * np.eye(H1.shape[1])
    HHG = np.linalg.solve(HH1, G1.T)
    kerH1 = np.dot(G1, HHG)
    kerH1 = (kerH1 + kerH1.T) / 2
    m1 = kerH1.shape[0]
    e3 = -np.ones(m1)
    
    vlb = np.zeros((m1, 1))
    vub = d1 * (np.ones((m1, 1)))
    G = np.vstack((np.eye(m1), -np.eye(m1)))
    h = np.vstack((vub, -vlb))
    alpha1 = solvers.qp(matrix(kerH1, tc='d'), matrix(e3, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'))
    alphasol1 = np.array(alpha1['x'])
    z = -np.dot(HHG, alphasol1)
    w1 = z[:len(z)-1]
    b1 = z[len(z)-1]

    # 2nd QPP
    QQ = np.dot(G1.T, G1)
    QQ = QQ + eps2 * np.eye(QQ.shape[1])
    QQP = np.linalg.solve(QQ, H1.T)
    kerH2 = np.dot(H1, QQP)
    kerH2 = (kerH2 + kerH2.T) / 2
    m2 = kerH2.shape[0]
    e4 = -np.ones(m2)
    
    vlb = np.zeros((m2, 1))
    vub = d2 * (np.ones((m2, 1)))
    cd = np.vstack((np.identity(m2), -np.identity(m2)))
    vcd = np.vstack((vub, -vlb))
    alpha2 = solvers.qp(matrix(kerH2, tc='d'), matrix(e4, tc='d'), matrix(cd, tc='d'), matrix(vcd, tc='d'))
    alphasol2 = np.array(alpha2['x'])
    z = np.dot(QQP, alphasol2)
    w2 = z[:len(z)-1]
    b2 = z[len(z)-1]

    # Test phase
    P_1 = TestX[:, :-1]
    y1 = np.dot(P_1, w1) + b1
    y2 = np.dot(P_1, w2) + b2
    
    Predict_Y = np.zeros((y1.shape[0], 1))
    for i in range(y1.shape[0]):
        if np.min([np.abs(y1[i]), np.abs(y2[i])]) == np.abs(y1[i]):
            Predict_Y[i] = 1
        else:
            Predict_Y[i] = 0
        
    no_test, no_col = TestX.shape
    err = 0.0
    Predict_Y = Predict_Y.T
    obs1 = TestX[:, no_col - 1]
    for i in range(no_test):
        if np.sign(Predict_Y[0, i]) != np.sign(obs1[i]):
            err += 1
    acc = ((TestX.shape[0] - err) / TestX.shape[0]) * 100
    return acc

# Step 5: Evaluate the model
def evaluate_model(clf, X_test, y_test):
    # Predict the labels for the test set
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

if __name__ == '__main__':
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Generate Hidden Matrix (using RVFL)
    X_train_transformed = generate_hidden_matrix(X_train)
    X_test_transformed = generate_hidden_matrix(X_test)
    
    # Combine the transformed data with labels for TWSVM
    DataTrain = np.hstack((X_train_transformed, y_train.values.reshape(-1, 1)))
    TestX = np.hstack((X_test_transformed, y_test.values.reshape(-1, 1)))
    
    # Train TWSVM
    accuracy = TWSVM_main(DataTrain, TestX, 0.5, 0.5)
    
    print(f'Accuracy: {accuracy:.2f}%')
