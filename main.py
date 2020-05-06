from CSVReader import CSVReader
from Regressor import SKLearnRegressor,MyRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from utils import MSE


#region **INITIALIZE**
reader=CSVReader("data/world-happiness-report-2017.csv")

#region **CAST ALL DATA TO FLOAT(3dplot pre.)**
for i in range(len(reader.testData["Freedom"])):
    reader.testData["Freedom"][i]=float(reader.testData["Freedom"][i])
for i in range(len(reader.testData["Economy..GDP.per.Capita."])):
    reader.testData["Economy..GDP.per.Capita."][i]=float(reader.testData["Economy..GDP.per.Capita."][i])
for i in range(len(reader.testData["Happiness.Score"])):
    reader.testData["Happiness.Score"][i]=float(reader.testData["Happiness.Score"][i])
#endregion

#endregion

#region **TRAIN**

trainDataFrame=pd.DataFrame.from_dict(reader.trainData)#get train data into a dataframe
testDataFrame=pd.DataFrame.from_dict(reader.testData)#get test data into a dataframe


myRegressor=MyRegressor()
myRegressor.fit(trainDataFrame[['Economy..GDP.per.Capita.', 'Freedom']], trainDataFrame['Happiness.Score'])
w0,w1,w2= myRegressor.intercept_, myRegressor.coef_[0], myRegressor.coef_[1]


skRegressor=SKLearnRegressor()
skRegressor.fit(trainDataFrame[['Economy..GDP.per.Capita.', 'Freedom']], trainDataFrame['Happiness.Score'])
W0,W1,W2= skRegressor.intercept_, myRegressor.coef_[0], myRegressor.coef_[1]



computedTestOutputsMyRegressor=myRegressor.predict(testDataFrame[['Economy..GDP.per.Capita.', 'Freedom']])
computedTestOutputsSKRegressor=skRegressor.predict(testDataFrame[['Economy..GDP.per.Capita.', 'Freedom']])
#endregion



#region **MEAN SQUARED ERROR**

errorMyRegressor=MSE(computedTestOutputsMyRegressor, reader.testData["Happiness.Score"])
errorSKRegressor=mean_squared_error(computedTestOutputsSKRegressor, reader.testData["Happiness.Score"])
print("MyRegressor error(MSE): "+str(errorMyRegressor))
print("SKRegressor error(MSE): "+str(errorSKRegressor))
#print("Difference is : "+str(errorMyRegressor-errorSKRegressor))
#endregion


#region **PLOT DATA**
ppp=plt.axes(projection='3d')

ppp.scatter3D(reader.testData["Freedom"][:10],reader.testData["Economy..GDP.per.Capita."][:10],reader.testData["Happiness.Score"][:10],label='real',depthshade=0)
ppp.scatter3D(reader.testData["Freedom"][:10],reader.testData["Economy..GDP.per.Capita."][:10], computedTestOutputsMyRegressor[:10], label='computed',depthshade=0)
ppp.set_xlabel('Freedom')
ppp.set_ylabel('GDP')
ppp.set_zlabel('Happiness Score')
ppp.set_edgecolors = ppp.set_facecolors = lambda *args:None

plt.legend()
plt.show()
#endregion