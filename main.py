#!/usr/bin/python3
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

def readFile(filePath):
  dataset = pd.read_csv(filePath)
  return dataset

def preProcessData(data):
  # Creating labelEncoder object
  le = preprocessing.LabelEncoder()
  # Converting string labels into numbers
  return le.fit_transform(data)

def testPreProcessedData(data, data_encoded):
  # View data with encoding
  dataView = zip(data, data_encoded)
  print(list(dataView))
  # View unique values
  print(set(dataView))

if __name__ == "__main__":
  # Read dataset from csv file
  dataset = readFile("assets/golf.csv")
  print(dataset)
  
  # Split dataset by columns
  outlook = dataset["Outlook"]
  temperature = dataset["Temperature"]
  humidity = dataset["Humidity"]
  windy = dataset["Windy"]
  label = dataset["Play"]

  # Pre process the columns
  outlook_encoded = preProcessData(outlook)
  temperature_encoded = preProcessData(temperature)
  humidity_encoded = preProcessData(humidity)
  windy_encoded = preProcessData(windy)
  label_encoded = preProcessData(label)
 
  print("===Testing Data===")

  # Test Data pre processed
  testPreProcessedData(outlook, outlook_encoded)
  testPreProcessedData(temperature, temperature_encoded)
  testPreProcessedData(humidity, humidity_encoded)
  testPreProcessedData(windy, windy_encoded)
  testPreProcessedData(label, label_encoded) 

  # Combine features to create rows
  features = list(zip(outlook_encoded, temperature_encoded, humidity_encoded, windy_encoded))
  
  print("===Features===")

  # Output to the screen
  print(features)

  # Naives Bayes implementation
#  test_size = 0.3 # means 70% for training and 30% for test
#  random_state = 109
  
  model = GaussianNB()
  X_train, X_test, y_train, y_test = train_test_split(features, label_encoded, test_size=0.3, random_state=109)

  # Train the model using the training sets
  model.fit(X_train, y_train)

  print("")
  print("")
  print("===Predictions===")
  
  # Get predicitons on test data
  y_pred = model.predict(X_test)
  print(y_pred)

  print("===Accuracy===")

  # Find accuracy of the model
  # Model accuracy, how often is the classifier correct?
  print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
  
