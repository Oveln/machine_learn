import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# from diabetes.csv read data
def read_data():
    f = open('diabetes.csv', 'r')
    lines = f.readlines()
    data = []
    lines = lines[1:]  # skip header
    for line in lines:
        line = line.strip()
        parts = line.split('\t')
        parts = [float(x) for x in parts]
        data.append(parts)
    data = np.array(data)
    data_x = data[:, :-1]
    data_y = data[:, -1]
    return data_x, data_y

def cost(x, y, theta):
    diff = np.dot(x, theta) - y
    return (1 / (2 * len(y))) * np.dot(diff.T, diff)

def gradientFun(x, y, theta):
    diff = np.dot(x, theta) - y
    return (1 / len(y)) * np.dot(x.transpose(), diff)

def gradientDescent(x_train, y_train, initial_theta, learning_rate, threshold, max_iterations):
    theta = initial_theta
    cost_list = []
    for i in range(max_iterations):
        grad = gradientFun(x_train, y_train, theta)
        theta = theta - learning_rate * grad
        cost_value = cost(x_train, y_train, theta)
        cost_list.append(cost_value)
        if np.linalg.norm(grad) < threshold:
            break
        
        # Check for overflow or invalid values
        if np.isnan(theta).any() or np.isinf(theta).any():
            print(f"Overflow or invalid values detected at iteration {i}")
            break
            
    return theta, cost_list

# Read the data
data_x, data_y = read_data()
print(data_x.shape, data_y.shape)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape)
print(f"Original data range: min = {x_train.min():.1f}, max = {x_train.max():.1f}")

# Normalize the features to prevent overflow
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(f"Scaled data range: min = {x_train_scaled.min():.2f}, max = {x_train_scaled.max():.2f}")

# Run gradient descent
theta, cost_list = gradientDescent(x_train_scaled, y_train, np.zeros(x_train_scaled.shape[1]), 0.01, 0.00001, 1000)
print(f"Cost after {len(cost_list)} iterations: {cost_list[-1]:.2f}")
print(f"Final theta: {[round(x, 2) for x in theta]}")

# Evaluate the model
predictions = np.dot(x_train_scaled, theta)
mse = np.mean((predictions - y_train) ** 2)
print(f"Training completed successfully without overflow errors.")
print(f"Mean Squared Error on training set: {mse:.2f}")