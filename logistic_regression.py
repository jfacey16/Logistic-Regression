import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def preprocess():
	data = pd.read_csv('weatherAUS.csv')

	# Drop certain features any any data with null values
	data = data.drop(['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date'],axis=1)
	data = data.dropna(how='any')

	# Change labels
	data['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
	data['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

	# Change categorical data to integers
	categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']
	data = pd.get_dummies(data, columns=categorical_columns)

	# standardize data set
	scaler = preprocessing.MinMaxScaler()
	scaler.fit(data)
	data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)

	y = data.pop('RainTomorrow')
	X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
	return X_train, X_test, y_train, y_test

def gradient_descent(X, p, y):
    """Returns gradient descent updates for the weights"""
    return np.dot(X.T, (p - y)) / y.shape[0]

def sigmoid(t):
    """Implementation of the sigmoid function"""
    return 1 / (1 + np.exp(-t))

def prediction(p):
    """Returns class predicitons based on logistic regression inputs"""
    return p >= 0.5

def cost_loss(p, y):
    """Log loss function for the regression"""
    return (-y * np.log(p) - (1 - y) * np.log(1 - p)).mean()

class LogisticRegression(object):

    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = preprocess()
        self.learn_rate = .01
        self.weights = np.zeros(self.X_train.shape[1])
        self.epochs = 10000

    def evaluate(self, X_test, y_test):
        """Returns a numpy array of prediction labels"""
        p = sigmoid(np.dot(X_test, self.weights))
        predict = prediction(p)
        return (predict == y_test).mean()

    def train(self, X_train, y_train):
        """"Sets the weights of the regression by training on input data"""
        self.weights = np.zeros(X_train.shape[1])
        
        for i in range(self.epochs):
            # get prediction
            p = sigmoid(np.dot(X_train, self.weights))
            # print loss
            if (i % 100 == 0):
                print('loss: ' + str(cost_loss(p,y_train)))
            # update weights
            self.weights -= self.learn_rate * gradient_descent(X_train, p, y_train)
