import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Sebastian Raschka's Perceptron implementation.
class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.

    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        
    def _shuffle(self, X, y):  # new
        """Shuffle training data"""  # new
        r = np.random.permutation(len(y))  # new
        return X[r], y[r]  # new
        
#Perceptron algorithm used on data that converges.  
ppn = Perceptron(eta=0.1, n_iter=20)
ppn.fit(np.array([[1.],[2.],[4.],[8.],[16.],[32.],[64.],[128.],[256.],[512.]]),
        np.array([-1,-1,-1,-1,-1,1,1,1,1,1])) #My own 1-d array

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

plt.tight_layout()
plt.show()
plt.savefig('converge_true.png', dpi=300)

#Perceptron algorithm used on data that does NOT converge.
ppn2 = Perceptron(eta=0.1, n_iter=50)
ppn2.fit(np.array([[1.],[2.],[3.],[5.],[8.],[13.],[21.],[34.],[55.],[89.]]),
        np.array([-1,1,-1,-1,1,1,-1,-1,1,1])) 

plt.plot(range(1, len(ppn2.errors_) + 1), ppn2.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

plt.tight_layout()
plt.show()
plt.savefig('converge_false.png', dpi=300)

#Sebastian Raschka's Adaline implementation
class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.

    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)

#Reading in the titanic data into usable numpy arrays.
titanicData = pd.read_csv("C:\Users\Abdul\Documents\comp_379\\train.csv")
survived = titanicData.iloc[0:,1].values
survived = np.where(survived == 1, 1, -1)

trainData = titanicData.iloc[0:,[2,4,5,6,7,9]].values
gender = titanicData.iloc[0:,4].values
gender = np.where(gender == 'male', 1, 0)
trainData[0:,1] = gender

trainData = trainData.astype(float)
invalid = (np.isnan(trainData)).sum(1)
trainData = trainData[invalid == 0, :]
survived = survived[invalid == 0]

testData = trainData[(int)(trainData.shape[0] * 0.7):,:] #First 70% of data
trainData = trainData[0:(int)(trainData.shape[0] * 0.7),:] #Last 70% of data

#Fit the data
adaTitanic = AdalineGD(n_iter=100, eta=0.0000006).fit(trainData, survived[0:trainData.shape[0]])

plt.plot(range(1, len(adaTitanic.cost_) + 1), np.log10(adaTitanic.cost_), marker='o')
plt.xlabel('Epochs')
plt.ylabel('log(Sum-squared-error)')
plt.tight_layout()
plt.show()
plt.savefig('adaline.png', dpi=300)

matchCount = 0.

for row in range(0,testData.shape[0]):
    if adaTitanic.predict(testData[row,:]) == survived[trainData.shape[0] + row]:
        matchCount += 1
        
print matchCount / testData.shape[0] #Rate of correct prediction

print adaTitanic.w_ #Gives me final weights


