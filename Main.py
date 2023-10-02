import pandas as pd
import numpy as np
import Functions
import Plotting 
from sklearn.metrics import mean_squared_error

#Raw data (Train and Test) on past sea levels is collected as a csv file (source=tidesandcurrents.noaa.gov)
train_data=pd.read_csv(r'C:\Users\meera\Documents\BTech CSE spl. AIML\VS code\Hackathon\SeaLvls_Train.csv')
test_data=pd.read_csv(r'C:\Users\meera\Documents\BTech CSE spl. AIML\VS code\Hackathon\SeaLvls_Test.csv')

Plotting.plot_the_data(train_data)

#Normalizing the data
train_data=Functions.scale(train_data)
test_data=Functions.scale(test_data)

#shuffle train set
shuffled_train_data=train_data.reindex(np.random.permutation(train_data.index))

trdata = shuffled_train_data[['Year', 'Month', 'Monthly_MSL']]
Year = trdata['Year'].values.reshape(-1,1)
Month = trdata['Month'].values.reshape(-1,1)
Monthly_MSL = trdata['Monthly_MSL'].values.reshape(-1,1)
TimeFrame = np.column_stack((trdata['Year'],trdata['Month']))

tsdata = test_data[['Year', 'Month', 'Monthly_MSL']]
Test_Year = tsdata['Year'].values.reshape(-1,1)
Test_Month = tsdata['Month'].values.reshape(-1,1)
Test_Monthly_MSL = tsdata['Monthly_MSL'].values.reshape(-1,1)
Test_TimeFrame = np.column_stack((tsdata['Year'],tsdata['Month']))

#hyperparameters
learning_rate = 0.001
epochs = 120       
batch_size = 24
validation_split = 0.2

#Invoke the functions to build and train the model.
my_model = Functions.build_model(learning_rate)
epochs, rmse, history = Functions.train_model(my_model, TimeFrame, Monthly_MSL, epochs, batch_size, validation_split)
Plotting.plot_the_loss_curve(epochs, history["root_mean_squared_error"], history["val_root_mean_squared_error"])

#Verifying model with test
results = my_model.evaluate(Test_TimeFrame, Test_Monthly_MSL, batch_size=batch_size)
print(results)

#Model predicts rising levels for the next fifteen years from 2015
pred_data=Functions.PredData()
Pred_TimeFrame = np.column_stack((pred_data['Year'],pred_data['Month']))
prediction=my_model.predict(Pred_Timeframe)
pred_data["Monthly_MSL"]=prediction
pred_data["Date"]=pd.to_datetime(pred_data[['Year', 'Month']].assign(day=1))
print(pred_data)
pred_data.to_csv(r'C:\Users\meera\Documents\BTech CSE spl. AIML\VS code\Hackathon\SeaLvls_Predicted.csv',index=False)

