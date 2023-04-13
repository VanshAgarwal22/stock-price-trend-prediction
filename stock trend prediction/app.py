import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


start = '2010-01-01'
end = '2019-12-31'

st.title('stock trend prediction')
user_input  = st.text_input('enter the Stock Ticker', 'AAPL')
df= data.DataReader(user_input, 'yahoo', start ,end)


#describe the data

st.subheader('Data from 2010 - 2019')
st.write(df.describe())

#visualiation
st.subheader('Closing price vs Time chat')
fig =plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Data from 2010 - 2019')
st.write(df.describe())


st.subheader('Closing price vs Time chat 100MA')
ma100 = df.Close.rolling(100).mean()
fig =plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)



st.subheader('Closing price vs Time chat 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig =plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200, 'b')
plt.plot(df.Close, 'g')
st.pyplot(fig)


#spliting the data

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int (len(df))])
print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler      #### for LSTM model we have to scale down the data between 0 and 1 , clsing prices ka data 0 na d1 k beech
scaler = MinMaxScaler(feature_range=(0,1)) 

data_training_array = scaler.fit_transform(data_training)       




#load my model

model = load_model('keras_model.h5')
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing , ignore_index=True)
input_data =scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100 : i])
  y_test.append(input_data[i , 0])

  x_test , y_test = np.array(x_test) , np.array(y_test)
  y_predicted = model.predict(x_test)

scaler.scale_
scale_factor = 1/[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor



#final graph

st.subheader('prediction vs original')
fig2 =plt.figure(figsize=(12,6))
plt.plot(y_test , 'b' , label='Original Price')
plt.plot(y_predicted , 'r' , label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Pirce')
plt.legend()
plt.show()
st.pyplot(fig2)