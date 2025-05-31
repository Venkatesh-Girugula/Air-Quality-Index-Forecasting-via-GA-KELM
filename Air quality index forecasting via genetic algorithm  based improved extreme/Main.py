from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from GAKELM import GeneticELMRegressor #import genetic algorithm based ELM
from sklearn_extensions.extreme_learning_machines import elm
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM #class for LSTM training
import os
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional #class for bidirectional LSTM as BILSTM
import math
from sklearn import svm


main = tkinter.Tk()
main.title("Air Quality ") #designing main screen
main.geometry("1300x1200")

global filename, dataset, X_train, X_test, y_train, y_test, X, Y, scaler, pca
global accuracy, precision, recall, fscore, values,cnn_model,algorithm, predict, test_labels,extension
mse = []
rmse = []

sc1 = MinMaxScaler(feature_range = (0, 1)) #use to normalize training data
sc2 = MinMaxScaler(feature_range = (0, 1)) #use to normalize label data

#interpolate function to deal with missing values and outliers
def interpolate_nans(X):
    """Overwrite NaNs with column value interpolations."""
    for j in range(X.shape[1]):
        mask_j = np.isnan(X[:,j])
        X[mask_j,j] = np.interp(np.flatnonzero(mask_j), np.flatnonzero(~mask_j), X[~mask_j,j])
    return X

def uploadDataset():
    global filename, dataset, labels, values
    filename = filedialog.askopenfilename(initialdir = "Dataset/Dataset.csv")
    text.delete('1.0', END)
    text.insert(END,'Dataset loaded\n\n')
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    text.insert(END,str(dataset))
    plt.figure(figsize=(10,4), dpi=100)
    plt.plot(dataset.Date[0:30], dataset.SO2[0:30], color='tab:red')
    plt.gca().set(title="Datewise SO2 Air Quality", xlabel='Date', ylabel="SO2 Values")
    plt.xticks(rotation=90)
    plt.show()


def processDataset():
    global dataset, X, Y
    global X_train, X_test, y_train, y_test, pca, scaler
    text.delete('1.0', END)
    dataset.fillna(0, inplace = True)
    Y = dataset.values[:,2:3]
    dataset.drop(['City'], axis = 1,inplace=True) #removing irrelevant columns
    dataset.drop(['Date'], axis = 1,inplace=True)
    dataset.drop(['PM2.5'], axis = 1,inplace=True)
    dataset.drop(['AQI_Bucket'], axis = 1,inplace=True)
    dataset = dataset.values
    X = dataset[:,3:dataset.shape[1]-1]

    #outlier and missing values removal using interpolation
    X = interpolate_nans(X)

    X = sc1.fit_transform(X)
    Y = sc2.fit_transform(Y)
    text.insert(END,"Normalized Training Features"+"\n")
    text.insert(END,X)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    text.insert(END,"Total records found in dataset  = "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset = "+str(X.shape[1])+"\n")
    text.insert(END,"80% dataset for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset for testing  : "+str(X_test.shape[0])+"\n")

def calculateMetrics(algorithm, predict, test_labels):
    predict = sc2.inverse_transform(np.abs(predict))
    test_label = sc2.inverse_transform(test_labels)
    predict = predict.ravel()
    test_label = test_label.ravel()
    mse_value = mean_squared_error(test_label, predict) / 100
    rmse_value = math.sqrt(mse_value)
    mse.append(mse_value)
    rmse.append(rmse_value)
    text.insert(END,algorithm+" MSE  : "+str(mse_value)+"\n")
    text.insert(END,algorithm+" RMSE : "+str(rmse_value)+"\n")
    plt.plot(test_label, color = 'red', label = 'Original Air Quality PM10')
    plt.plot(predict, color = 'green', label = 'Predicted Air Quality PM10')
    plt.title(algorithm+' Air Quality Prediction')
    plt.xlabel('Test Data')
    plt.ylabel('Predicted Air Quality')
    plt.legend()
    plt.show()

def trainSVM():
    global X_train, y_train, X_test, y_test
    global algorithm, predict, test_labels
    text.delete('1.0', END)
    svm_cls = svm.SVR()
    svm_cls.fit(X_train, y_train.ravel())
    predict = svm_cls.predict(X_test)
    predict = predict.reshape(-1, 1)
    calculateMetrics("Existing SVR", predict, y_test)

def trainGEML():
    global X_train, y_train, X_test, y_test
    global algorithm, predict, test_labels
    text.delete('1.0', END)
    elm = GeneticELMRegressor()
    elm.fit(X_train, y_train)#now train genetic elm on training data
    predict = elm.predict(X_test) #perform prediction on test data
    calculateMetrics("Propose GA-KELM", predict, y_test)#calculate metrics

def trainLSTM():
    global X_train, y_train, X_test, y_test
    global algorithm, predict, test_labels,extension
    text.delete('1.0', END)
    X_train1 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test1 = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #now train extension BI-LSTM algorithm
    extension = Sequential()
    #adding BILSTM 
    extension.add(Bidirectional(LSTM(units = 50, input_shape = (X_train1.shape[1], X_train1.shape[2]), return_sequences=True)))
    extension.add(Dropout(0.5))
    #adding another layer to filter data
    extension.add(Bidirectional(LSTM(units = 50, return_sequences = True)))
    extension.add(Dropout(0.5))
    #adding another layer
    extension.add(Bidirectional(LSTM(units = 50, return_sequences = True)))
    extension.add(Dropout(0.5))
    extension.add(Bidirectional(LSTM(units = 50)))
    extension.add(Dropout(0.5))
    extension.add(Dense(units = 1))
    extension.compile(optimizer = 'adam', loss = 'mean_squared_error')
    if os.path.exists('model/extension_weights.hdf5') == False:
        model_check_point = ModelCheckpoint(filepath='model/extension_weights.hdf5', verbose = 1, save_best_only = True)
        extension.fit(X_train1, y_train, epochs = 250, batch_size = 4, validation_data=(X_test1, y_test), callbacks=[model_check_point], verbose=1)
    else:
        extension = load_model('model/extension_weights.hdf5')
    predict = extension.predict(X_test1)
    calculateMetrics("Extension Bi-LSTM", predict, y_test)

def graph():
    global X_train, y_train, X_test, y_test
    global algorithm, predict, test_labels
    text.delete('1.0', END)
    df = pd.DataFrame([['SVR','MSE',mse[0]],['SVR','RMSE',rmse[0]],
                       ['Propose GA-KELM','MSE',mse[1]],['Propose GA-KELM','RMSE',rmse[1]],
                       ['Extension BI-LSTM','MSE',mse[2]],['Extension BI-LSTM','RMSE',rmse[2]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.title("Existing SVM, Propose GA-KELM & Extension BI-LSTM Performance Graph")
    plt.show()

def predict():
    global X_train, y_train, X_test, y_test
    global algorithm, predict, test_labels,extension
    text.delete('1.0', END)
    dataset = pd.read_csv("Dataset/testData.csv")
    dataset.fillna(0, inplace = True)
    dataset.drop(['City'], axis = 1,inplace=True) #removing irrelevant columns
    dataset.drop(['Date'], axis = 1,inplace=True)
    dataset.drop(['PM2.5'], axis = 1,inplace=True)
    dataset.drop(['AQI_Bucket'], axis = 1,inplace=True)
    dataset = dataset.values
    X = dataset[:,3:dataset.shape[1]-1]
    #outlier and missing values removal using interpolation
    X = interpolate_nans(X)
    X1 = sc1.transform(X)
    X1 = np.reshape(X1, (X1.shape[0], X1.shape[1], 1))
    predict = extension.predict(X1) #perform air quality prediction using extension BI-LSTM extension object
    predict = sc2.inverse_transform(predict)
    for i in range(len(predict)):
        text.insert(END,"Test Data = "+str(X[i])+"=====> Predicted Air Quality : "+str(predict[i,0])+"\n")
    


font = ('times', 16, 'bold')
title = Label(main, text='Air Quality Index Forecasting via Genetic Algorithm-Based Improved Extreme Learning Machine')
title.config(bg='gray24', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=27,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

processButton = Button(main, text="Preprocess & Split Dataset", command=processDataset)
processButton.place(x=250,y=100)
processButton.config(font=font1)

svmButton = Button(main, text="Run Existing SVR", command=trainSVM)
svmButton.place(x=490,y=100)
svmButton.config(font=font1)

gemlButton = Button(main, text="Run Propose GA-KELM", command=trainGEML)
gemlButton.place(x=730,y=100)
gemlButton.config(font=font1)

lstmButton = Button(main, text="Run Extension BILSTM", command=trainLSTM)
lstmButton.place(x=970,y=100)
lstmButton.config(font=font1)

graphButton = Button(main, text="Comparision Graph", command=graph)
graphButton.place(x=10,y=150)
graphButton.config(font=font1)

predict = Button(main, text="Predict Air Quality", command=predict)
predict.place(x=250,y=150)
predict.config(font=font1)


main.config(bg='peach puff')
main.mainloop()
