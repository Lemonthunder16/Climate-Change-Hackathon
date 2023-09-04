import tensorflow as tf
import pandas as pd

#Pre Processing to be done on the data set (Train and Test)
def scale(data):
    scale_factor=10
    data["Monthly_MSL"]/=scale_factor
    return data

#Functions to build and train the model
def build_model(my_learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=10, input_dim=2))
    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

def train_model(model, X, Y, my_epochs, my_batch_size=None, my_validation_split=0.1):
    history = model.fit(x=X, y=Y, batch_size=my_batch_size, epochs=my_epochs, validation_split=my_validation_split)
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]
    return epochs, rmse, history.history

#Function to download predicted values
def PredData():
    Year=[i for i in range(2015,2031) for j in range(12)]
    Month=[i for j in range(16) for i in range(1,13)]
    pred_data=pd.DataFrame({"Year":Year,"Month":Month})
    pred_data['Date']=pred_data['Year']+((pred_data['Month']-1)/12)
    pred_data["Monthly_MSL"]=0
    return pred_data