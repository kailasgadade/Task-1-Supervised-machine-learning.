# # Task 1: Prediction Using Supervised ML.
# # Author: Gadade Kailas Rayappa.
# # Step-1 import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_excel("C:\\Users\\Kailas\\OneDrive\\Desktop\\Data I.xlsx")
data
# # Step -2 Exploring the data
data.shape
data.describe()
data.info()
# # Step-3 Data Visualiztion 
data.plot(kind='scatter',x='hours',y='scores');
plt.show()
data.corr(method='pearson')
data.corr(method='spearman')
hours=data['hours']
scores=data['scores']
sns.distplot(hours)
sns.distplot(scores)
# # Step-4 Linear Regression
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=50)
from sklearn.linear_model import LinearRegression
reg= LinearRegression()
reg.fit(x_train, y_train)
m=reg.coef_
c=reg.intercept_
line=m*x+c
plt.scatter(x,y)
plt.plot(x,line);
plt.show()
y_pred=reg.predict(x_test)
actual_predicted=pd.DataFrame({'Target':y_test,'predicted':y_pred})
actual_predicted
sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()
h=9.25
s=reg.predict([[h]])
print("If student studies for {} hours per day he/she will score {}% in exam.".format(h,s))
# # Step-5 Model Evolution
from sklearn import metrics
from sklearn.metrics import r2_score
print('mean absolute error:',metrics.mean_absolute_error(y_test,y_pred))
print('R2 score:',r2_score(y_test,y_pred))
# # Thank You! 
