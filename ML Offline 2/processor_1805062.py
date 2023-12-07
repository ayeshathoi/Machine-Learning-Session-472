import pandas as panda 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler  
from sklearn.preprocessing import OneHotEncoder

from adaptive_boosing_1805062 import *
from Performance_Calculator_1805062 import *

class Processor:

    def test_and_train(X_train, Y_train, X_test, Y_test,alpha,epoch):
        
        # without adaboost
        print("MINIBATCH GRADIENT DESCENT TRAINING WITHOUT ADABOOST")
        print("#################################################################")
        print("TRAINING...................")
        hypotheses, hypotheses_weights = adaboost(X_train, Y_train,1,alpha,epoch)
        y_hat = weighted_majority_predict(X_train, hypotheses, hypotheses_weights)
        performance_metrics(y_hat, Y_train )
        print("TRAINING DONE")
        print("TESTING...................")
        y_hat = weighted_majority_predict(X_test, hypotheses, hypotheses_weights)
        performance_metrics(y_hat, Y_test)
        print("TESTING DONE\n")


        print("#################################################################")
        

        ## ADABOOST
        print("Adaboost TRAINING")
        start = 5
        stop = 21
        step = 5
        print("TRAINING...................")
        for rounds in range(start,stop,step):
            hypotheses, hypotheses_weights = adaboost(X_train, Y_train,rounds,alpha,epoch)
            y_hat = weighted_majority_predict(X_train, hypotheses, hypotheses_weights)
            performance_metrics(y_hat, Y_train )
            print("\nTRAINING DONE FOR ROUND ",rounds)

            print("TESTING...................")
            y_hat = weighted_majority_predict(X_test, hypotheses, hypotheses_weights)
            performance_metrics(y_hat, Y_test)
            print("\nTESTING DONE FOR ROUND ",rounds)
            print("######################################################################")
        print("Adaboost TRAINING DONE")

    ##############################################################################################################

    def entropy(y):
        if len(y) == 0:
            return 0
        
        #calculate Boolean B for output
        B = np.sum(y) / len(y)
        if B == 0 or B == 1:
            return 0
        
        #calculate entropy
        entropy = -B * np.log2(B) - (1 - B) * np.log2(1 - B)
        return entropy


    def info_gain(X, y, feature_index, threshold):
        
        if len(np.unique(X[:,feature_index])) > 2:
            P = np.sum(X[:,feature_index] >= threshold)
            N = np.sum(X[:,feature_index] < threshold)
            Probability_P = P / (P + N)
            Probability_N = N / (P + N)

            y_P = y[X[:,feature_index] >= threshold]
            y_N = y[X[:,feature_index] < threshold]

        else :
            P = np.sum(X[:,feature_index] == 1)
            N = np.sum(X[:,feature_index] == 0)
            Probability_P = P / (P + N)
            Probability_N = N / (P + N)
            y_P = y[X[:,feature_index] == 1]
            y_N = y[X[:,feature_index] == 0]

        entropy_P = Processor.entropy(y_P)
        entropy_N = Processor.entropy(y_N)

        remainder_entropy = Probability_P * entropy_P + Probability_N * entropy_N

        return Processor.entropy(y) - remainder_entropy


    def selection(X_train, X_test, Y_train, Y_test,feature_count):
        info_gains = []

        # if X_feature continuous
        for i in range(X_train.shape[1]):
            #check if X is real or categorical
            if len(np.unique(X_train[:,i])) > 2:
                # if X_feature continuous
                threshold = np.median(X_train[:,i])
                info_gains.append(Processor.info_gain(X_train, Y_train, i, threshold))
            else:
                # if X_feature categorical only 0 and 1
                info_gains.append(Processor.info_gain(X_train, Y_train, i, 0))

        # arg sort returns the indices that would sort an array
        best_features = np.argsort(info_gains)[-feature_count:]
        #print the index of the best features

        X_train = X_train[:, best_features]
        X_test = X_test[:, best_features]

        return X_train, X_test, Y_train, Y_test
    ##############################################################################################################

    #Telco Dataset
    def Telco_preprocessing(input_file,alpha,epoch,feature_count):

        df = panda.read_csv(input_file)
        df = df.to_numpy()
        df = df[1:,1:]
        df = panda.DataFrame(df)

        #split the data into train and test

        #convert to matrix 
        df= df.replace(" ", np.nan)
        df = df.dropna()
        df.replace({'No phone service': 'No'}, inplace=True)
        df.replace({'No internet service': 'No'}, inplace=True)
        df[4] = panda.to_numeric(df[4], errors='coerce')
        df[17] = panda.to_numeric(df[17], errors='coerce')
        df[18] = panda.to_numeric(df[18], errors='coerce')

        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
        train_data = panda.DataFrame(train_data)
        test_data = panda.DataFrame(test_data)

        #use simple imputer to fill the missing values
        numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        
        fitter = numeric_imputer.fit(train_data[[4,17,18]])
        train_data[[4,17,18]] = fitter.transform(train_data[[4,17,18]])
        test_data[[4,17,18]] = fitter.transform(test_data[[4,17,18]])

        # TRAIN DATA
        # normalize the data
        scaler = MinMaxScaler()
        scaler_fitter = scaler.fit(train_data[[4,17,18]])
        train_data_float = scaler_fitter.transform(train_data[[4,17,18]])

        #use one hot encoder to encode the categorical data,select categorical features
        cata_features = [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,19]

        onehotencoder = OneHotEncoder(drop='if_binary').fit(train_data[cata_features])
        train_data_non_float = onehotencoder.transform(train_data[cata_features]).toarray()
        train_data = np.concatenate((train_data_float, train_data_non_float), axis=1)

        # TEST DATA

        test_data_float = scaler_fitter.transform(test_data[[4,17,18]])
        test_data_non_float = onehotencoder.transform(test_data[cata_features]).toarray()
        test_data = np.concatenate((test_data_float, test_data_non_float), axis=1)

        # transform the test data
        X_train = train_data[:, :-1]
        Y_train = train_data[:, -1]
        X_test = test_data[:, :-1]
        Y_test = test_data[:, -1]
        # print(X_train)
        # print(Y_train)
        # print(X_test)
        # print(Y_test)

        # feature selection
        print("DATASET 1 :TELCO DATASET")
        print("#################################################################")

        X_train, X_test, Y_train, Y_test = Processor.selection(X_train, X_test, Y_train, Y_test,feature_count)
        return Processor.test_and_train(X_train, Y_train, X_test, Y_test,alpha,epoch)
    


    ##############################################################################################################

    def adultPreprocessing(input_train_file,input_test_file,alpha,epoch,feature_count):
        #adult dataset .data file read
        train_data = panda.read_csv(input_train_file, header=None)
        test_data = panda.read_csv(input_test_file, header=None, skiprows=1)

        #convert to matrix
        train_data = train_data.to_numpy()
        test_data = test_data.to_numpy()

        train_data = panda.DataFrame(train_data)
        test_data = panda.DataFrame(test_data)

        train_data.replace({' <=50K': 0, ' >50K': 1}, inplace=True)
        test_data.replace({' <=50K.': 0, ' >50K.': 1}, inplace=True)


        train_data = train_data[~np.isin(train_data, [' ?']).any(axis=1)]
        test_data = test_data[~np.isin(test_data, [' ?']).any(axis=1)]

        #use simple imputer to fill the missing values
        numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        fitter = numeric_imputer.fit(train_data[[0,2,4,10,11,12,14]])
        train_data[[0,2,4,10,11,12,14]] = fitter.transform(train_data[[0,2,4,10,11,12,14]])
        test_data[[0,2,4,10,11,12,14]] = fitter.transform(test_data[[0,2,4,10,11,12,14]])

        # TRAIN DATA
        # normalize the data
        scaler = MinMaxScaler()
        scaler_fitter = scaler.fit(train_data[[0,2,4,10,11,12,14]])
        train_data_float = scaler_fitter.transform(train_data[[0,2,4,10,11,12,14]])

        #use one hot encoder to encode the categorical data,select categorical features
        cata_features = [1,3,5,6,7,8,9,13]
        onehotencoder = OneHotEncoder(drop='if_binary').fit(train_data[cata_features])
        train_data_non_float = onehotencoder.transform(train_data[cata_features]).toarray()
        train_data = np.concatenate(( train_data_non_float,train_data_float), axis=1)

        # TEST DATA
        test_data_float = scaler_fitter.transform(test_data[[0,2,4,10,11,12,14]])
        test_data_non_float = onehotencoder.transform(test_data[cata_features]).toarray()
        test_data = np.concatenate((test_data_non_float,test_data_float), axis=1)

        # transform the test data
        X_train = train_data[:, :-1]
        Y_train = train_data[:, -1]
        X_test = test_data[:, :-1]
        Y_test = test_data[:, -1]


        print("DATASET 2 :ADULT DATASET")
        print("#################################################################")
        

        # feature selection
        X_train, X_test, Y_train, Y_test = Processor.selection(X_train, X_test, Y_train, Y_test,feature_count)
        return Processor.test_and_train(X_train, Y_train, X_test, Y_test,alpha,epoch)

        

    ##############################################################################################################
    def creditCardPreprocessing(input_file,alpha,epoch,feature_count):
        df = panda.read_csv('./Datasets/Dataset3/creditcard.csv')
        df = df.to_numpy()
        df = df[1:,1:]
        df = panda.DataFrame(df)

        #convert to matrix 
        df= df.replace(" ", np.nan)
        df = df.dropna()
        ### Balancing the dataset
        df_class_0 = df[df[29] == 0]
        df_class_1 = df[df[29] == 1]

        #smaller subset of class 0
        df_class_0 = df_class_0.sample(n=20000, random_state=42)

        # split the data
        train_data_0, test_data_0 = train_test_split(df_class_0, test_size=0.2, random_state=42)
        train_data_1, test_data_1 = train_test_split(df_class_1, test_size=0.2, random_state=42)

        train_data = panda.concat([train_data_0, train_data_1])
        test_data = panda.concat([test_data_0, test_data_1])
    
        #use simple imputer to fill the missing values
        numeric_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

        fitter = numeric_imputer.fit(train_data)
        train_data = fitter.transform(train_data)
        test_data = fitter.transform(test_data)

        # TRAIN DATA
        # normalize the data
        scaler = MinMaxScaler()
        scaler_fitter = scaler.fit(train_data)
        X_train = scaler_fitter.transform(train_data)
        X_test = scaler_fitter.transform(test_data) 

        X_train = train_data[:, :-1]
        Y_train = train_data[:, -1]
        X_test = test_data[:, :-1]
        Y_test = test_data[:, -1]   

        print("DATASET 3 :CREDIT CARD DATASET")
        print("#################################################################")

        X_train, X_test, Y_train, Y_test = Processor.selection(X_train, X_test, Y_train, Y_test,feature_count)
        return Processor.test_and_train(X_train, Y_train, X_test, Y_test,alpha,epoch)
