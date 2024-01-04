import torchvision.datasets as ds
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt
epsilon = 1e-6
##################################################3
#best_model = model2_.0009.pickle
def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
##############################################

###################################################
### Loss Function ###
def Loss_cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred+epsilon)) 

############################################################################
# F1 score, accuracy, confusion matrix #
def calculation(y_pred, y_validation, isValidation = False):
    # confusion matrix
    # Loss calculation
    Loss = Loss_cross_entropy(y_validation, y_pred)
    confusion_matrix = np.zeros((26,26))
    for i in range(len(y_pred)):
        confusion_matrix[y_pred[i]][y_validation[i]] += 1 
    print("LABELS               TP      FP      FN      TN  ")
    F1_score = np.zeros(26)
    
    for i in range(26):
        TP = confusion_matrix[i][i]
        FP = np.sum(confusion_matrix[i]) - TP
        FN = np.sum(confusion_matrix[:,i]) - TP
        TN = np.sum(confusion_matrix) - TP - FP - FN
        print("{:10} {:10} {:10} {:10} {:10}".format(chr(i + 65), TP, FP, FN, TN))
        F1_score[i] = 2 * TP / (2 * TP + FP + FN)
    #print("F1 score: ", F1_score)
    f1_macro = np.mean(F1_score)
    print("F1 macro: ", f1_macro)
    accuracy = np.sum(y_pred == y_validation) / len(y_validation)
    print("accuracy: ", accuracy)


########################################################################
class BatchNormalization:
    def __init__(self,input_node):
        self.gamma = np.random.randn(1, input_node)
        self.beta = np.random.randn(1, input_node)

    def forward(self, x):
        self.variance = np.sqrt(np.var(x, axis=0, keepdims=True) + epsilon)
        mean = np.mean(x, axis=0, keepdims=True)
        self.x_normalized = (x - mean) / self.variance
        out = self.gamma * self.x_normalized + self.beta

        return out

    def backward(self, grad_output, learning_rate, adam_flag):
        grad_input = grad_output * self.gamma  / self.variance
        grad_gamma = np.sum(grad_output * self.x_normalized, axis=0, keepdims=True)
        grad_beta = np.sum(grad_output, axis=0, keepdims=True)
        self.beta -= learning_rate * grad_beta
        self.gamma -= learning_rate * grad_gamma
        return grad_input 
    
    def clear(self):
        self.gamma = None
        self.beta = None
        self.x_normalized = None
        self.variance = None
############################################################################

### Activation Layer, output node = input node

# 1. RELU
class RELU():
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        return np.maximum(0,input) 

    def backward(self, output_gradient, learning_rate, adam_flag):
        return output_gradient * (self.input > 0)
    
    def clear(self):
        self.input = None

# 2. Softmax
class SOFTMAX():
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        exp = np.exp(input - np.max(input, axis=1).reshape(-1, 1))
        self.out = exp / np.sum(exp, axis=1).reshape(-1, 1)
        return self.out 

    def backward(self, output_gradient, learning_rate, adam_flag):
        return output_gradient * self.out * (1 - self.out)
    
    def clear(self):
        self.input = None
        self.out = None


  
# 3. DROP OUT
class DROPOUT():
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def forward(self, input):
        self.dropout_mask = np.random.binomial(1, self.dropout_rate, size=input.shape) / self.dropout_rate
        return input * self.dropout_mask

    def backward(self, output_gradient,learning_rate, adam_flag):
        return output_gradient * self.dropout_mask

    def clear(self):
        self.dropout_mask = None
        self.dropout_rate = None
    

##########################################################################
class DENSELAYER():
    def __init__(self,input_node, output_node):
        self.variance = np.sqrt(2.0 / (input_node + output_node))
        self.weights = np.random.randn(input_node, output_node) * self.variance # xavier initialization
        self.biases = np.random.randn(output_node,1) * self.variance
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.m = np.zeros_like(self.weights)
        self.v = np.zeros_like(self.weights)
        self.t = 0

    def Adam(self,learning_rate, gradient):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return m_hat / (np.sqrt(v_hat) + epsilon)

    def forward(self, input):
        self.input = input
        
        multiplication = np.dot(input,self.weights) + self.biases.T
        return multiplication 

    def backward(self, output_gradient, learning_rate,adam_flag):
        grad_input = np.dot(output_gradient, self.weights.T)
        grad_weights = np.dot(self.input.T,output_gradient)
        grad_biases = output_gradient.sum(axis=0)

        # Adam
        # add grad_bias to grad_weights
        if adam_flag == True:
            grad_weights= self.Adam(learning_rate, grad_weights)

        self.weights -= learning_rate * grad_weights
        grad_biases = grad_biases.reshape(-1,1) # reshape for broadcasting as grad_biases is 1D array
        self.biases -= learning_rate * grad_biases 
        return grad_input
    
    def clear(self):
        self.m = None
        self.v = None
        self.input = None
        self.output = None

################################################################################################################

# Feed Forward Neural Network #
        
class FeedForwardNeuralNetwork():
    def predict(self, data):
        for layer in self.layers:       
            data = layer.forward(data)
        return data


################################################################################################################

# read data
        
def read_data():
    train_validation_dataset = ds.EMNIST(root='./data', split='letters',train=True,transform=transforms.ToTensor(),download=False)
    independent_test_dataset = ds.EMNIST(root='./data',split='letters',train=False,transform=transforms.ToTensor())
    # split the train_validation_dataset into train and validation dataset 85-15
    train_dataset, validation_dataset = train_test_split(train_validation_dataset, test_size=0.15, random_state=42)
    return train_dataset, validation_dataset, independent_test_dataset

# preprocessing

def preprocessing():
    train_dataset, validation_dataset, independent_test_dataset = read_data()
    # np.nan remove
    train_dataset = [x for x in train_dataset if not np.isnan(x[0]).any()]
    validation_dataset = [x for x in validation_dataset if not np.isnan(x[0]).any()]
    independent_test_dataset = [x for x in independent_test_dataset if not np.isnan(x[0]).any()]

    # each train and validation dataset has 26 classes
    train_x = np.array([np.array(x[0].flatten()) for x in train_dataset])
    train_y = np.array([np.array(x[1]) for x in train_dataset])
    validation_x = np.array([np.array(x[0].flatten()) for x in validation_dataset])
    validation_y = np.array([np.array(x[1]) for x in validation_dataset])
    test_x = np.array([np.array(x[0].flatten()) for x in independent_test_dataset])
    test_y = np.array([np.array(x[1]) for x in independent_test_dataset])


    # one hot encoding
    oneHotEncoder = OneHotEncoder().fit(train_y.reshape(-1, 1))
    train_y = oneHotEncoder.transform(train_y.reshape(-1, 1)).toarray()
    validation_y = oneHotEncoder.transform(validation_y.reshape(-1, 1)).toarray()
    test_y = oneHotEncoder.transform(test_y.reshape(-1, 1)).toarray()
    
    return train_x, train_y, validation_x, validation_y, test_x, test_y

################################################################################################################

class model2():

    def independent_test(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        print("Test Loss: {}".format(Loss_cross_entropy(y_test, y_pred)))
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1) 
        calculation(y_pred, y_test,False)

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

x_train, y_train, x_validation, y_validation, x_test, y_test = preprocessing()
model = pickle_load('model_1805062.pickle')
model.independent_test(x_test, y_test)
