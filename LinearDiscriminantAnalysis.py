# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:04:10 2019

@author: lukes - done as a one man team
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
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from scipy.stats import multivariate_normal
from scipy.spatial.distance import mahalanobis
"""
LDA class works to create linear discriminants to transform
the data to a new plane that best separates the data
"""

class LDA:

    def __init__(self):
        pass
    
    """
    Creates a dx1 mean vector for the entire dataset
    
    d: number of features in dataset
    """
    def dataset_mean_vector(self):
        return np.mean(self.data, axis=0)
    
    """
    Creates a dxc matrix of mean vectors for each class
    
    d:number of features in dataset
    c:number of classes in dataset
    """
    def class_mean_matrix(self):
        data = self.data
        target = self.target
        classes = np.unique(target)
        mean_matrix = p.DataFrame(columns = classes)
        for i in range(len(classes)):
            mean_matrix.loc[:,classes[i]] = np.mean(data[target==i], axis=0)           
        return mean_matrix

    """
    Creates a dxd between scatter matrix
    - scatter between class means and the overall dataset mean
    
    d:number of features in dataset
    """
    def between_class_scatter(self):
        mean = self.dataset_mean
        data = self.data
        class_mean_matrix = self.class_mean
        target = self.target
        dimension = len(class_mean_matrix.index)
        scatter_b = np.zeros((dimension,dimension)) 
        classes = np.unique(target)
        for i in range(len(classes)):
            class_size = len(data[target==i])
            class_mean = class_mean_matrix.loc[:,i].values
            scatter_value = class_size*((class_mean - mean).reshape(-1,1)*((mean - class_mean)))
            scatter_b += scatter_value        
        return -scatter_b
    
    """
    Creates a dxd within class scatter matrix
    - scatter between class mean and data points within the class.
        Individual scall scatter matrices determined and summed together
        
    
    d:numver of features in dataset
    """    
    def within_class_scatter(self, shrinkage):
        data = self.data
        class_mean_matrix = self.class_mean
        target = self.target
        dimension = data.shape[1]
        scatter_w = np.zeros((dimension,dimension))        
        classes = np.unique(target)
        for j in range(len(classes)):
            class_data = data[target==j]
            class_mean = class_mean_matrix.loc[:,j].values
            for i in range(len(class_data)):
                sample = class_data[i]
                scatter_value = ((sample - class_mean).reshape(-1,1)*((class_mean - sample)))
                scatter_w += scatter_value    
            scatter_w = scatter_w/(len(class_data))
            
        return (1-shrinkage)*(-scatter_w)+shrinkage*np.identity(data.shape[1])
    
    """
    Generate Linear Discriminants from the Eigenvectors in the Matrix inv(SW)*SB
    
    -The eigenvectors with the max eigenvalues will create discriminants with the best
        seperability
    """
    def discriminant_generator(self):
        within_scatter = self.scatter_w
        between_scatter = self.scatter_b
        matrix = (np.linalg.inv(within_scatter)).dot(between_scatter)
        eigValues, eigVectors= np.linalg.eig(matrix)
        eigVectors = eigVectors.transpose()
        indices = eigValues.argsort()[::-1]
        eigVectors = eigVectors[indices]
        eigValues = eigValues[indices]
        return eigVectors, eigValues
    
    """
    The previously determined Linear Discriminants are used to transform the data
    to the new plane or axis
    
    """
    def fit_transform(self,data, dimension=1):
#        data = data.values
        eig = self.eig
        eigVectors = eig[0].real
        eigValues = eig[1].real 
        
#        indices = eigValues.argsort()[-dimension:][::-1]
#        lda_vec = eigVectors[indices]
#        lda_vec = lda_vec.transpose()
#        print(indices)
#        self.lda_vec = lda_vec
#        new_feature_plane = data.dot(lda_vec)
        new_feature_plane = data.dot(eigVectors.transpose())
        return new_feature_plane
        
    """
    The classes in the dataset should be represented by multivariate Gaussian
    distribution where each dimensions represents a feature in the data.
    
    The pdf determines the "likilihood" that a point belongs to a class.
    """
    def multivariate_gaussian_pdf(self, mean, cov, sample):
        
        #covariance = self.scatter_w/(len(self.data))
        covariance = cov
        dimension = len(self.class_mean.index)     
        pi_pow = math.pow(2*math.pi, dimension)
        det_cov = np.linalg.det(covariance)
        inv_cov = np.linalg.inv(covariance)
        
        smpl_men = (sample-mean)
        smpl_men_t = (sample-mean).reshape(1,dimension)
        exp_val = math.exp(-.5*smpl_men_t.dot(inv_cov).dot(smpl_men))
        val = (exp_val/(math.sqrt(pi_pow*det_cov)))
        
        return val
        
    """
    Classification stage of the Algorithm
    """
    def generate_predictions(self, input_data):
        
        data = self.data
        target = self.target
        class_mean = self.class_mean
        class_proportions= np.array([])
        
        classes = np.unique(target)

        for i in range(len(classes)):
            class_proportions = np.append(class_proportions, len(data[target==i])/len(data))

        cov = self.scatter_w
                
        class_pdfs = np.array([])
        for i in range(len(classes)):
            class_pdf = self.multivariate_gaussian_pdf(class_mean.loc[:,i].values,cov,input_data.values)
            class_pdfs = np.append(class_pdfs, class_pdf)
                

        """
        Score function, Not working but not a necessity for the classifier
        """        
#        score0 = (mean0)*(inv_cov*(x.transpose()) - 1/2*(mean0.transpose())*(inv_cov)*(mean0))+math.log(p0)
#        score1 = (mean1)*(inv_cov*(x.transpose()) - 1/2*(mean1.transpose())*(inv_cov)*(mean1))+math.log(p1)
#        score2 = (mean2)*(inv_cov*(x.transpose()) - 1/2*(mean2.transpose())*(inv_cov)*(mean2))+math.log(p2)
                    
        """
        The posterior probability is determined for each class as
        -> PDF_classK*priorK/sum for all classes(PDF_classN*priorN)
        """
        total_prob = 0
        for i in range(len(classes)):
            total_prob += class_proportions[i]*class_pdfs[i]
        
        class_posterior_probs = np.array([])
        for i in range(len(classes)):
            posteriorProb = class_proportions[i]*class_pdfs[i]/total_prob
            class_posterior_probs = np.append(class_posterior_probs, posteriorProb)
        
        
        
#        print("\nCLASS 0 ", class_posterior_probs[0])
#        print("CLASS 1 ", class_posterior_probs[1])
#        print("CLASS 2 ", class_posterior_probs[2])
        
        
        return class_posterior_probs.argsort()[-1:][::-1], class_posterior_probs
    
    
    def boundary_gen(self):
        
        t = np.linspace(0,30)
        inv_cov = np.linalg.inv(self.scatter_w)
        
        mean0 = self.class_mean[0].values
        mean1 = self.class_mean[1].values
        
        mean0_1 = mean0-mean1
        mean1_0 = mean1-mean0
        mean0_1_T = mean0_1.T
        
        inv_cov_diff = inv_cov.dot(mean1_0)
        inv_cov_diff_T = inv_cov_diff.T
        
        mahalDist = mean0_1_T.dot(inv_cov).dot(mean0_1)
        
        boundary = np.array([])
        
        for i in range(len(t)):
            val = 2*inv_cov_diff_T*t[i]+mahalDist
            boundary = np.append(boundary, val)
        return boundary
    
    
    """
    Build model is called to first train the classifier before transformation
    -Creates Mean Vector
    -Creates Class Mean Matrix
    -Genereates Scatter between matrix
    -Generates Scatter within matrix
    -EigenVectors and EigenValues determined
    """
    def build_model(self, data, target, shrinkage=0):
        self.data = data.values
        self.target = target.values
        self.dataset_mean = self.dataset_mean_vector()
        self.class_mean = self.class_mean_matrix()
        self.scatter_b = self.between_class_scatter()
        self.scatter_w = self.within_class_scatter(shrinkage)
        self.eig = tuple(self.discriminant_generator())
        