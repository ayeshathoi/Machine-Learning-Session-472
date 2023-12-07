import numpy as np
import warnings
warnings.filterwarnings('ignore')

class Learner: 


    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def predict(X, w):
        data_size = X.shape[0]
        X = np.concatenate((np.ones((data_size, 1)), X), axis=1)
        y_hat = Learner.sigmoid(np.dot(X, w))
        for i in range(len(y_hat)):
            if y_hat[i] >= 0.5:
                y_hat[i] = 1
            else:
                y_hat[i] = 0

        return np.array(y_hat).reshape(data_size, 1)

    def gradient(X, y_predicted, y_true,data_weights):
        error = y_predicted - y_true
        for i in range(len(error)):
            error[i][0] = error[i][0] * data_weights[i]
        dw = np.dot(X.T, error)
        return dw

    def train(X, y_true, learning_rate, epochs,data_weights):
        data_size, feature_size = X.shape
        
        X = np.concatenate((np.ones((data_size, 1)), X), axis=1) 
        w = np.random.randn(feature_size + 1, 1) # w is a column vector

        y_true = y_true.reshape(data_size, 1)

        changer = learning_rate/epochs

        X_mini = np.array_split(X, epochs) #split the data into mini batches
        y_mini = np.array_split(y_true, epochs) #split the data into mini batches
        data_weights_mini = np.array_split(data_weights, epochs) #split the data into mini batches
        for epoch in range(epochs):
            y_predicted = Learner.sigmoid(np.dot(X_mini[epoch], w))
            dw = Learner.gradient(X_mini[epoch], y_predicted, y_mini[epoch],data_weights_mini[epoch])
            
            w = w - learning_rate * dw
            learning_rate = learning_rate - changer

        return w
