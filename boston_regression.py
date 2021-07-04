from pandas.core.algorithms import mode
import tensorflow as tf
from tensorflow import keras
import tensorflow
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import activations
from tensorflow.python.keras.backend import relu
from tensorflow.keras.optimizers import RMSprop

df = pd.read_csv(
    "https://raw.githubusercontent.com/dphi-official/Datasets/master/Boston_Housing/Training_set_boston.csv")
print(df.head())

X = df.drop('MEDV', 1)
y = df.MEDV

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.3, random_state=42)

n_features = X.shape[1]
print(n_features)

model = Sequential([
    Dense(20, activation='relu', input_shape=(n_features,)),
    Dense(15, activation='relu'),
    Dense(1)
])

# compiling the model
# we are going to use the RMSprop as our optimizer here
optimizer = RMSprop(learning_rate=0.002)
model.compile(optimizer, loss='mean_squared_error')
# fit the model to  the training sets
model.fit(X_train, y_train, epochs=50, batch_size=30, verbose=1)

# evaluation of the model
print("MSE of the model is: ", model.evaluate(X_test, y_test))
