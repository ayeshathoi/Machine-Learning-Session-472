import torchvision.datasets as ds
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt



epsilon = 1e-6
##################################################################
losses_epoch            = []
accuracy_model          = []
losses_learningRate     = []
f1_macro_model          = []
###################################################################



#### pickle ####

def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
##############################################
    
# plot the graph #
    
def plotting(x, y, x_label, y_label, title):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # save the graph
    plt.savefig(title + ".png")

###################################################
### Loss Function ###
def Loss_cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred+epsilon)) 

def MSE(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def Loss_cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true

def MSE_derivative(y_true, y_pred):
    return -2 * (y_true - y_pred)
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

    if isValidation == True:
        f1_macro_model.append(f1_macro)
        accuracy_model.append(accuracy)
        losses_learningRate.append(Loss)


########################################################################
class BatchNormalization:
    def __init__(self):
        self.gamma = 1
        self.beta = 0

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
        pass
    
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
    def __init__(self,input_node, output_node):
        np.random.seed(1)
        self.input_node = input_node
        self.output_node = output_node
        self.layers = []

    def predict(self, data):
        for layer in self.layers:       
            data = layer.forward(data)
        return data
    
    def train_prop(self, learning_rate,output_data, feature_data, epoch_num, mini_batch_size, adam_flag):
        for epoch in range(epoch_num):
            # shuffle the data
            shuffle_index = np.random.permutation(len(feature_data))
            feature_data = feature_data[shuffle_index]
            output_data = output_data[shuffle_index]

            for i in range(0, len(feature_data), mini_batch_size):
                mini_batch_feature_data = feature_data[i:i+mini_batch_size]
                mini_batch_output_data = output_data[i:i+mini_batch_size]      
                # Batch Normalization

                predicted_output = self.predict(mini_batch_feature_data)
                grad_loss = Loss_cross_entropy_derivative(mini_batch_output_data, predicted_output)              
                
                for layer in reversed(self.layers):
                    grad_loss = layer.backward(grad_loss, learning_rate, adam_flag)

            predicted_output = self.predict(feature_data)
            Loss = Loss_cross_entropy(output_data, predicted_output)
            print("Epoch: ", epoch, "Loss: ", Loss)
            losses_epoch.append(Loss)
            learning_rate *= 0.95

    def clear(self):
        for layer in self.layers:
            layer.clear()

################################################################################################################

# read data
        
def read_data():
    train_validation_dataset = ds.EMNIST(root='./data', split='letters',train=True,transform=transforms.ToTensor(),download=True)
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

# models design #

class model1():
    def __init__(self, input_node, output_node,learning_rate, epoch_num, mini_batch_size):
        self.input_node = input_node
        self.output_node = output_node
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        self.mini_batch_size = mini_batch_size
        self.layers = []

    def model_design(self):
        
        self.model = FeedForwardNeuralNetwork(self.input_node, self.output_node)
        # add layer
        self.model.layers.append(DENSELAYER(input_node, 512))
        self.model.layers.append(RELU())
        self.model.layers.append(DROPOUT(0.8))
        self.model.layers.append(DENSELAYER(512, 128))
        self.model.layers.append(RELU())
        self.model.layers.append(DROPOUT(0.9))
        self.model.layers.append(RELU())
        self.model.layers.append(DENSELAYER(128, output_node))
        self.model.layers.append(RELU())
        self.model.layers.append(SOFTMAX())
        return self.model
    
    def model_train(self, x_train, y_train, adam_flag):
        self.model.train_prop(self.learning_rate, y_train, x_train, self.epoch_num, self.mini_batch_size, adam_flag)
        # pop drop out layer
        self.model.layers.pop(2)
        self.model.layers.pop(4)

    def test(self, x_train, y_train, x_validation, y_validation):

        y_pred = self.model.predict(x_train)
        print("Training Loss: {}".format(Loss_cross_entropy(y_train, y_pred)))
        y_pred = np.argmax(y_pred, axis=1)
        y_train = np.argmax(y_train, axis=1)
        calculation(y_pred, y_train,False)

        y_pred = self.model.predict(x_validation)
        print("Validation Loss: {}".format(Loss_cross_entropy(y_validation, y_pred)))
        y_pred = np.argmax(y_pred, axis=1)
        y_validation = np.argmax(y_validation, axis=1) 
        calculation(y_pred, y_validation,True)
    
    def clear(self):
        self.model.clear()

class model2():
    def __init__(self, input_node, output_node, learning_rate, epoch_num, mini_batch_size):
        self.input_node = input_node
        self.output_node = output_node
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        self.mini_batch_size = mini_batch_size
        self.layers = []

    def model_design(self):
        
        self.model = FeedForwardNeuralNetwork(self.input_node, self.output_node)
        # add layers
        self.model.layers.append(DENSELAYER(input_node, 1024))
        self.model.layers.append(RELU())
        self.model.layers.append(DROPOUT(0.8))
        self.model.layers.append(DENSELAYER(1024, 128))
        self.model.layers.append(RELU())
        self.model.layers.append(DROPOUT(0.5))
        self.model.layers.append(DENSELAYER(128, output_node))
        self.model.layers.append(RELU())
        self.model.layers.append(SOFTMAX())
        return self.model
    
    def model_train(self, x_train, y_train, adam_flag):
        self.model.train_prop(self.learning_rate, y_train, x_train, self.epoch_num, self.mini_batch_size, adam_flag)
        # pop drop out layer
        self.model.layers.pop(2)
        self.model.layers.pop(4)

    def test(self,x_train, y_train, x_validation, y_validation):

        y_pred = self.model.predict(x_train)
        print("Training Loss: {}".format(Loss_cross_entropy(y_train, y_pred)))
        y_pred = np.argmax(y_pred, axis=1)
        y_train = np.argmax(y_train, axis=1)
        calculation(y_pred, y_train,False)

        y_pred = self.model.predict(x_validation)
        print("Validation Loss: {}".format(Loss_cross_entropy(y_validation, y_pred)))
        y_pred = np.argmax(y_pred, axis=1)
        y_validation = np.argmax(y_validation, axis=1) 
        calculation(y_pred, y_validation,True)


    def clear(self):
        self.model.clear()


class model3():
    def __init__(self, input_node, output_node, learning_rate, epoch_num, mini_batch_size):
        self.input_node = input_node
        self.output_node = output_node
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        self.mini_batch_size = mini_batch_size
        self.layers = []

    def model_design(self):
        
        self.model = FeedForwardNeuralNetwork(self.input_node, self.output_node)
        # add layers
        self.model.layers.append(BatchNormalization())
        self.model.layers.append(DENSELAYER(input_node, 600))
        self.model.layers.append(RELU())
        self.model.layers.append(DROPOUT(0.8))
        self.model.layers.append(DENSELAYER(600, output_node))
        self.model.layers.append(RELU())
        self.model.layers.append(SOFTMAX())
        return self.model
    
    def model_train(self, x_train, y_train, adam_flag):
        self.model.train_prop(self.learning_rate, y_train, x_train, self.epoch_num, self.mini_batch_size, adam_flag)
        # pop drop out layer
        self.model.layers.pop(3)


    def test(self,x_train, y_train, x_validation, y_validation):

        y_pred = self.model.predict(x_train)
        print("Training Loss: {}".format(Loss_cross_entropy(y_train, y_pred)))
        y_pred = np.argmax(y_pred, axis=1)
        y_train = np.argmax(y_train, axis=1)
        calculation(y_pred, y_train,False)

        y_pred = self.model.predict(x_validation)
        print("Validation Loss: {}".format(Loss_cross_entropy(y_validation, y_pred)))
        y_pred = np.argmax(y_pred, axis=1)
        y_validation = np.argmax(y_validation, axis=1) 
        calculation(y_pred, y_validation,True)

    def clear(self):
        self.model.clear()

################################################################################################################


# # read data
x_train, y_train, x_validation, y_validation, x_test, y_test = preprocessing()

# main input__num
input_node = x_train.shape[1]
# main output__num
output_node = y_train.shape[1]

learning_rate = [5e-3, 2e-3, 9e-4, 5e-4]
epoch_num = 100
mini_batch_size = 512

#######################################################################################################
adam_flag = False
max_accuracy = 0
best_model = None
best_loss = []
for i in learning_rate:
    model = model1(input_node, output_node, i, epoch_num, mini_batch_size)
    model.model_design()
    model.model_train(x_train, y_train,adam_flag)
    model.clear()
    pickle_save(model, "model1_" + str(i) + ".pickle")
    plotting(range(epoch_num), losses_epoch, "epoch", "loss", "model3_epoch vs loss")
    model.test(x_train, y_train, x_validation, y_validation)
    if max_accuracy < accuracy_model[-1]:
        max_accuracy = accuracy_model[-1]
        best_model = i
        best_loss = losses_epoch

    losses_epoch = []
    


print("best model: ", best_model)
# best model is model 2 with learning rate 0.005?


# # model 1
# accuracy_1 = [0.8845619658119658,0.798931623931624,0.8115384615384615,0.775534188034188]
# F1_score_1 = [0.8685450698451784,0.7728114098795993,0.7856659034425468,0.747627976666034]
# loss_1 = [7217.2961825964785,12562.862715095878,12222.020607903749,15235.668286959975]


# # model 2
# accuracy_2 = [0.9202457264957264,0.9259081196581197,0.9263945810883434,0.923138779115373]
# F1_score_2 = [0.9198677938807278,0.925661097039563,0.9266025641025641,0.923397435897436]
# loss_2 = [6231.741761775326,5154.27828379741,4611.488649866143,4647.204103429012]

# # model 3
# accuracy_3 = [0.841292735042735,0.9047542735042735,0.9044871794871795,0.8944444444444445]
# F1_score_3 = [0.8366523442902769,0.9044258262783036,0.9041684236403367,0.8941205387636473]
# loss3 = [13268.9239716886,6351.431457190261,6204.328582434878,6813.685061167433]
# learning_rate = [5e-3, 2e-3, 9e-4, 5e-4]
# plt.figure()
# # plot accuracy_1, accuracy_2, accuracy_3 in one graph with marker
# plt.plot(learning_rate, accuracy_1, marker='o', label='model 1')
# plt.plot(learning_rate, accuracy_2, marker='o', label='model 2')
# plt.plot(learning_rate, accuracy_3, marker='o', label='model 3')
# plt.legend()
# plt.xlabel('Learning Rate')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs LR')
# #save the plot as a file
# plt.savefig('Accuracy vs Learning Rate.png')
# plt.figure()
# # plot F1_score_1, F1_score_2, F1_score_3 in one graph with marker
# plt.plot(learning_rate, F1_score_1, marker='o', label='model 1')
# plt.plot(learning_rate, F1_score_2, marker='o', label='model 2')
# plt.plot(learning_rate, F1_score_3, marker='o', label='model 3')
# plt.legend()
# plt.xlabel('Learning Rate')
# plt.ylabel('F1_score')
# plt.title('F1_score vs LR')
# #save the plot as a file
# plt.savefig('F1_score vs Learning Rate.png')

# # clear
# plt.figure()
# # plot loss_1, loss_2, loss_3 in one graph with marker
# plt.plot(learning_rate, loss_1, marker='o', label='model 1')
# plt.plot(learning_rate, loss_2, marker='o', label='model 2')
# plt.plot(learning_rate, loss3, marker='o', label='model 3')
# plt.legend()
# plt.xlabel('Learning Rate')
# plt.ylabel('Loss')
# plt.title('Loss vs LR')
# #save the plot as a file
# plt.savefig('Loss vs Learning Rate.png')




