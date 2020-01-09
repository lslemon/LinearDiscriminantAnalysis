# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:30:19 2019

@author: lukes
"""
import sklearn
import numpy as np
import pandas as p
import scipy
import math
import random
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import colors

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LinearDiscrimAnalysis
from sklearn.metrics import confusion_matrix

import LinearDiscriminantAnalysis
from LinearDiscriminantAnalysis import LDA

def column_mapper(new_titles, old_titles):
    mapped_columns = {}
    for i in range(0, len(new_titles)):
        mapped_columns[old_titles[i]] = new_titles[i]    
    return mapped_columns

def convert_df(data):
    for i  in range(len(categories)-1):
        data.loc[:, categories[i]] = p.to_numeric(data.loc[:,categories[i]],errors='coerce')

def plot_distribution(data, target):
    
    plt.scatter(data[:,0], np.zeros_like(new_plane[:,1]),c=target.values)
    sns.distplot(data[:,0][target==0], hist=False, kde=True, color = 'yellow', label = 'Class 0')
    sns.distplot(data[:,0][target==1], hist=False, kde=True, color = 'purple', label = 'Class 1')
    sns.distplot(data[:,0][target==2], hist=False, kde=True, color = 'blue', label = 'Class 2')
    plt.title("LDA LUKE SLEMON")
    plt.legend()
    plt.show()

def plot_data(lda, transformed_data, transformed_point = None):
    
    plt.scatter(transformed_data[:,0][train_target==0],transformed_data[:,1][train_target==0], color = 'yellow', s=5, label = 'Class 0')
    plt.scatter(transformed_data[:,0][train_target==1],transformed_data[:,1][train_target==1], color = 'purple', s=5, label = 'Class 1')
    plt.scatter(transformed_data[:,0][train_target==2],transformed_data[:,1][train_target==2], color = 'blue', s=5, label = 'Class 2')
    new_mean = lda.fit_transform(lda.dataset_mean, 2)
    plt.scatter(new_mean[0],new_mean[1], color = 'red', marker="1", s=50)
    new_mean_class0 = lda.fit_transform(lda.class_mean[0], 2)
    plt.scatter(new_mean_class0[0],new_mean_class0[1], color = 'red', marker="1", s=50)
    new_mean_class1 = lda.fit_transform(lda.class_mean[1], 2)
    plt.scatter(new_mean_class1[0],new_mean_class1[1], color = 'red', marker="1",s=50)
    new_mean_class2 = lda.fit_transform(lda.class_mean[2], 2)
    plt.scatter(new_mean_class2[0],new_mean_class2[1], color = 'red', marker="1", s=50)
    if  transformed_point is not None:
        plt.scatter(transformed_point[0],transformed_point[1], color = 'green', marker="2", s=100)
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title("LDA LUKE SLEMON")
    plt.legend()
    plt.show()

    
"""
    Pre-processing data stage
"""
#Read the data
textFile = open("LDAResults.txt", "w")

categories = 'length,width,thickness,surface_area,mass,compactness,hardness,shell_top_radius,water_content,carbohydrate_content,variety'.split(',')
dataset = p.read_csv('hazlenut_dataset.csv')
dataset = dataset.transpose()
dataset.rename(columns=column_mapper(categories, dataset.columns), inplace=True)
column_mapper(categories, dataset.columns)
convert_df(dataset)
#Encode target values
le = LabelEncoder()
dataset.loc[ : ,'variety'] = le.fit_transform(dataset.loc[ : , 'variety'])
categories = categories[0:10]

dataset = dataset[dataset.variety!=2]
#Split target and sample data

target = dataset.loc[ : , 'variety']
data = dataset.loc[: , categories[0:10]]
#data = data.loc[:, categories[0:2]]

class0, class1, class2 = data[target==0], data[target==1], data[target==2]
target0, target1, target2 = target[target==0], target[target==1], target[target==2] 
#t = np.linspace(-20,20)

"""
    Classification Step
"""

lda = LDA()
clf = LinearDiscrimAnalysis()
accuracies1 = np.array([])
accuracies2 = np.array([])

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size = 1/3)

lda.build_model(data, target, shrinkage=.3)
new_plane = lda.fit_transform(data.values,2)
boundary = lda.boundary_gen()

x = np.array([])
for i in range(data.shape[1]):
    l = [np.min(data.values[:,i], axis=0), np.max(data.values[:,i], axis=0)]
    x = np.append(x,l)
    
mean0 = lda.class_mean[0].values
mean1 = lda.class_mean[1].values
scatw = lda.scatter_w

inv_scatw = np.linalg.inv(scatw)
means = mean1-mean0

p0 = len(class0)/len(data)
p1 = len(class1)/len(data)

ln = 2*math.log(p1/p0)
mw = inv_scatw.dot(means)
mahal = (mean0-mean1).reshape(1,-1).dot(inv_scatw).dot(mean0-mean1)


w = (mw)

bound = 2*x*w+mahal+ln

#plt.scatter(data.iloc[:,0],data.iloc[:,1], c=target)
#plt.plot(bound)
#plt.show()

bound_T = lda.fit_transform(bound, 2)
plt.scatter(new_plane[:,0],new_plane[:,1], c=target)
plt.plot(bound_T)
plt.show()


#sns.distplot(new_plane[:,0][target==0], hist=False, kde=True, color = 'yellow', label = 'Class 0')
#sns.distplot(new_plane[:,0][target==1], hist=False, kde=True, color = 'purple', label = 'Class 1')
#plt.show()
             
#for j in range(10):
#    textFile.write("\n")
#    textFile.write("Iteration "+str(j))
#    textFile.write("\n")
#    
#    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size = 1/3)
#    
#    clf = clf.fit(train_data, train_target)
#    lda.build_model(train_data, train_target, shrinkage=.3)
#    new_plane = lda.fit_transform(train_data.values,2)
##    plot_data(lda, new_plane)
#    
#    predictions1 = np.array([])
#    probabilities1 = np.array([])
#    predictions2 = np.array([])
#    probabilities2 = np.array([])
#    
#    
#    for i in range(len(test_data)):
#        prediction, probability = lda.generate_predictions(test_data.iloc[i])
#        predictions1 = np.append(predictions1, prediction)
#        probabilities1 = np.append(probabilities1, probability)
#        new_point = lda.fit_transform(test_data.iloc[i] ,2)    
#        plot_data(lda, new_plane, new_point)    
#    predictions2 = clf.predict(test_data)
#    
#    accuracy1 = np.mean(predictions1 == test_target)
#    accuracies1 = np.append(accuracies1, accuracy1)
#    
#    accuracy2 = np.mean(predictions2 == test_target)
#    accuracies2 = np.append(accuracies2, accuracy2)
#    print(confusion_matrix(test_target, predictions1))
#    textFile.write("Custom Implementation\n")
#    textFile.write("Confusion Matrix\n")
#    textFile.write(str(confusion_matrix(test_target, predictions1)))
#    textFile.write("\nAccuracy ")
#    textFile.write(str(accuracy1))
#    
#    textFile.write("\n\n")
#    textFile.write("SKLEARN Implementation\n")
#    textFile.write("Confusion Matrix\n")
#    textFile.write(str(confusion_matrix(test_target, predictions2)))
#    textFile.write("\nAccuracy ")
#    textFile.write(str(accuracy2))
#    textFile.write("\n")
#    
#    plot_distribution(new_plane, train_target)
    
#textFile.write("\n\n")

#print("Custom Impllementation ",np.mean(accuracies1))
#print("SKLearn Impllementation ",np.mean(accuracies2))
#textFile.write("Custom Implementation Avg Accuracy\n"+str(np.mean(accuracies1)))
#textFile.write("\n\n")
#textFile.write("SKLEARN Implementation Avg Accuracy\n"+str(np.mean(accuracies2)))
#textFile.close()