import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle


df = pd.read_csv (r"C:\Users\RAJAT AGARWAL\Desktop\proj sem 4\audit_data\trial.csv ")
df_temp= pd.DataFrame(df, columns= ['Sector_score'  , 'PARA_A' ,'PARA_B' ,'TOTAL' ,  'Money_Value' , 'Score'])
df_temp = df_temp.dropna()


X = df_temp[['Sector_score'  ,'PARA_A' ,'PARA_B' ,'TOTAL' ,  'Money_Value'  ]]
y = df_temp['Score']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X, y)





pickle.dump(regressor, open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))


  


print(model.predict([[3.89, 0, 1.1 , 1.1 , 0.007]]))

