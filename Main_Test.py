#Trying it out by encoding date as a floating point year

import pandas as pd
import numpy as np
import Functions
import Plotting 

#Raw data (Train and Test) on past sea levels is collected as a csv file (source=tidesandcurrents.noaa.gov)
train_data=pd.read_csv(r'C:\Users\meera\Documents\BTech CSE spl. AIML\VS code\Hackathon\SeaLvls_Train.csv')
test_data=pd.read_csv(r'C:\Users\meera\Documents\BTech CSE spl. AIML\VS code\Hackathon\SeaLvls_Test.csv')

Plotting.plot_the_data(train_data)

#Data is pre-processed to shortlist neccessary elements required for the regression model
train_data=Functions.pre_process(train_data)
test_data=Functions.pre_process(test_data)

#define feature and label
my_feature = "Date"             
my_label = "Monthly_MSL"

#shuffle train set
shuffled_train_data=train_data.reindex(np.random.permutation(train_data.index))

X_train=shuffled_train_data[my_feature]
Y_train=shuffled_train_data[my_label]
X_test=test_data[my_feature]
Y_test=test_data[my_label]


#hyperparameters
learning_rate = 0.001
epochs = 260       
batch_size = 32
validation_split = 0.2

#Invoke the functions to build and train the model.
my_model = Functions.build_model(learning_rate)
epochs, rmse, history = Functions.train_model(my_model, X_train, Y_train, epochs, batch_size, validation_split)
Plotting.plot_the_loss_curve(epochs, history["root_mean_squared_error"], history["val_root_mean_squared_error"])

#Verifying model with test
results = my_model.evaluate(X_test, Y_test, batch_size=batch_size)
print(results)

#Model predicts rising levels for the next fifteen years from 2015
pred_data=Functions.PredData()
prediction=my_model.predict(pred_data["Date"])
pred_data["Monthly_MSL"]=prediction
pred_data=pred_data.drop("Date",axis='columns')
pred_data["Date"]=pd.to_datetime(pred_data[['Year', 'Month']].assign(day=1))
print(pred_data)
pred_data.to_csv(r'C:\Users\meera\Documents\BTech CSE spl. AIML\VS code\Hackathon\SeaLvls_Predicted.csv',index=False)
