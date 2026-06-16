#This file simulates a production environment where you receive a new flower and want an immediate 
# prediction without training anything. 

import numpy as np
from model.model import FFN

deployed_model =  FFN()

# Inject the trained brain into it! our weights and biases
deployed_model.loadModel("best_iris_model.pkl")

new_flower = np.array([[5.1, 3.5, -1.4, 0.2]])

probabities = deployed_model.forward(new_flower)
predicted_class = np.argmax(probabities ,axis=1)[0]

classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
print(f"The model predicts this flower is a: {classes[predicted_class]}")



