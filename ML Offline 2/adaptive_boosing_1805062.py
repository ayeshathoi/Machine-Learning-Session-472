from model_1805062 import *

def adaboost(X, Y, rounds,alpha,epoch):

    dataset_size = X.shape[0]
    data_weight = np.full((dataset_size), 1 / dataset_size)
    hypotheses = []
    

    hypotheses_weights = []
    Y = Y.T
    for i in range(rounds):
        #resample
        # examples = np.concatenate((X, Y), axis=1)
        # data = examples[np.random.choice(dataset_size, size=dataset_size, replace=True, p=data_weight)]
        # X = data[:, :X.shape[1]]
        # Y = data[:, -1:]

        data_weights = np.array(data_weight).reshape(dataset_size,1)

        w = Learner.train(X, Y, alpha, epoch,data_weights)
 
        y_hat = Learner.predict(X, w) #predictions
        error = 0
        for j in range(dataset_size):
            if y_hat[j] != Y[j]:
                error += data_weight[j]
        if error <= 0.5:
            hypotheses.append(w)
            hypotheses_weights.append(np.log2((1 - error) / error))
        else :
            continue
        # calculate hypothesis weight
        for j in range(dataset_size):
            if y_hat[j] == Y[j]:
                data_weight[j] *= error / (1 - error)
        # normalize weights
        data_weight /= np.sum(data_weight)
        
    hypotheses_weights /= np.sum(hypotheses_weights)  

    return hypotheses, np.array(hypotheses_weights).reshape(len(hypotheses), 1)

def weighted_majority_predict(X, hypotheses, hypothesis_weights):
    Data_set_size = X.shape[0]
    hypotheses_number = len(hypotheses)
    
    X = np.concatenate((np.ones((Data_set_size, 1)),X), axis=1)

    y_hat = [] # list of predictions for each hypothesis
    threshold = 0.5

    for i in range(hypotheses_number):
        y_predicted = Learner.sigmoid(np.dot(X, hypotheses[i]))
        y_hat.append([1 if val >= threshold else -1 for val in y_predicted])
        
    y_hat = np.array(y_hat)
    
    # calculating weighted majority hypothesis and storing predictions
    weighted_majority_hypothesis = np.dot(y_hat.T, hypothesis_weights)
    y_hat = [1 if val >= 0 else 0 for val in weighted_majority_hypothesis]
    
    return np.array(y_hat).reshape(Data_set_size, 1)