from sklearn import linear_model
import pandas as pd
from numpy import linalg

class SKLearnRegressor:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = 0.0
        self.__regr=linear_model.LinearRegression()


    def fit(self, trainInputs, trainOutputs):

        self.__regr.fit(trainInputs,trainOutputs)
        self.intercept_,self.coef_=self.__regr.intercept_,self.__regr.coef_

    def predict(self, xValues):
        return self.__regr.predict(xValues)



class MyRegressor:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = None

    def fit(self, trainInputs, trainOutputs):
        X=trainInputs.values.tolist()
        for i in range(0,len(X)):
            X[i].insert(0,1)
        xTr = [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]
        Y=[[el] for el in trainOutputs.values.tolist()]
        inverse=linalg.inv(self.matrixMult(xTr,X)).tolist()

        pr=self.matrixMult(inverse,xTr)
        W=self.matrixMult(pr,Y)
        self.intercept_,self.coef_=W[0],W[1:]

        self.intercept_=self.intercept_[0]
        self.coef_=[el[0] for el in self.coef_]



    def matrixMult(self,A, B):
        '''

        :param A:matrix
        :param B: matrix
        :return: A*B
        '''
        return [[sum([float(A[i][m]) * float(B[m][j]) for m in range(len(A[0]))]) for j in range(len(B[0]))] for i in range(len(A))]

    def predict(self, xValues):
        xValues=xValues.values.tolist()

        if (isinstance(xValues[0], list)):
            return [self.intercept_ + self.coef_[0] * float(val[0])+self.coef_[1] * float(val[1]) for val in xValues]
        else:
            return [self.intercept_ + self.coef_[0] * float(val)+self.coef_[1] * float(val) for val in xValues]
