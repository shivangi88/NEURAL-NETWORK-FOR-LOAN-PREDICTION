# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:39:51 2019

@author: 19030431
"""

#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics

def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma


def str_to_int(df):
    str_columns = df.select_dtypes(['object']).columns
    print(str_columns)
    for col in str_columns:
        df[col] = df[col].astype('category')

    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df

def pre_processing(df):
    #min_max=MinMaxScaler()
    df.drop(delete_columns, axis=1, inplace=True)
    # Target, Embarked one-hot
    df = one_hot(df, df.loc[:, ["Target"]].columns)
    
    # String to int
    df = str_to_int(df)
    #df = scale(df[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
    return df


def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        del df[each]
        df = pd.concat([df, dummies], axis=1)
    return df

#Importing CSV Files:-
df_train = pd.read_csv('X_train.csv')
df_test = pd.read_csv('X_test.csv')
delete_columns = ["Loan_ID"]

#Preprocessing:-
df_train = pre_processing(df_train)
#save Loan_ID for evaluation
test_loan_id = df_test["Loan_ID"]
df_test = pre_processing(df_test)
#Splitting data into features and labels
features = df_train.iloc[:, 1:].values
features = feature_normalize(features)
labels = df_train.iloc[:, :1].values
print(features.shape, labels.shape)


train_x=df_train.values
train_x=train_x[:,:-1]
train_y=train_x[:,:1]

test_x = df_test.values
test_x=test_x[:,:-1]
test_y=test_x[:,:1]
print(test_x.shape, test_y.shape)
rnd_indices = np.random.rand(len(features))  < 0.80

#train_x = features[~rnd_indices]
#train_y = labels[~rnd_indices]


feature_count = train_x.shape[1]
label_count = train_y.shape[1]
print(feature_count, label_count)

#Shuffling:-
kf = KFold(n_splits=2, random_state=None, shuffle=True) # Define the split - into 2 folds 
kf.get_n_splits(train_x) # returns the number of splitting iterations in the cross-validator
print(kf) 
for train_index, test_index in kf.split(train_x):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_val = train_x[train_index], train_x[test_index]
    y_train, y_val = train_y[train_index], train_y[test_index]


# inputs
training_epochs = 2500
learning_rate = 0.01
hidden_layers = feature_count  - 1
cost_history = np.empty(shape=[1],dtype=float)

X = tf.placeholder(tf.float32,[None,feature_count])
Y = tf.placeholder(tf.float32,[None,label_count])

# models

initializer = tf.contrib.layers.xavier_initializer()
h0 = tf.layers.dense(X, hidden_layers, activation=tf.nn.relu, kernel_initializer=initializer)
#h0 = tf.nn.dropout(h0, 0.95)
h1 = tf.layers.dense(h0, label_count, activation=None)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=h1))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
predicted = tf.nn.sigmoid(h1)
correct_pred = tf.equal(tf.round(predicted), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# session

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(training_epochs + 1):
        #sess.run(optimizer, feed_dict={X: train_x, Y: train_y})
        loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict={
                                 X:X_train, Y: y_train})
        cost_history = np.append(cost_history, acc)
        if step % 500 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tTrainAcc: {:.2%}".format(
                step, loss, acc))
            
    # Test model and check accuracy
            
    for step in range(training_epochs + 1):
        loss,acc = sess.run([cost,accuracy], feed_dict={X: X_val, Y: y_val})
        #cost_history = np.append(cost_history, acc)
        if step % 500 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tCrossValAcc: {:.2%}".format(
                step, loss, acc))
    # Save test result
    test_predict_result = sess.run(tf.cast(tf.round(predicted), tf.int32), feed_dict={X: test_x})
    #Evaluating Results
    print('Test Accuracy:',metrics.accuracy_score(test_y, test_predict_result))
    unique, counts = np.unique(test_y, return_counts=True)
    print(dict(zip(unique, counts)))
    final_result=pd.DataFrame(columns=['Loan_ID','TargetNew','TargetOld'])
    final_result["Loan_ID"]=test_loan_id
    final_result["TargetNew"]=test_predict_result
    final_result.to_csv('result.csv')
    print('Confusion Matrix:')
    confusion=metrics.confusion_matrix(test_y, test_predict_result)
    print(confusion)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print('Accuracy Score:',metrics.accuracy_score(test_y, test_predict_result))    
    print('Classification Error:',1 - metrics.accuracy_score(test_y, test_predict_result))
    print('Sensitivity/Recall Score:',metrics.recall_score(test_y, test_predict_result))
    print('Specificity/Precision Score:',metrics.precision_score(test_y, test_predict_result))































