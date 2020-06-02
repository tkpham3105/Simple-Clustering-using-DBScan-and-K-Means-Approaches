# -*- coding: utf-8 -*-
"""

K_MEANS_CLUSTERING
@author: PHAM Trung Kien

"""

###############################################################################
# Import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


###############################################################################
# Change the working directory to where dataset is located
os.chdir("D:\\tkpham\\COMP4331\\Assignment_3")


###############################################################################
# Load the data file as dataframes using pandas
dataset = pd.read_csv("a3dataset.txt", sep = ',', names = ['X','Y'])


###############################################################################
'''
Basically, create a new column merging two corresponding columns of x and y coordinates
Also, create a new column to store the predicted cluster, initializing by 0 for every points 
For example:
    X  |  Y            X  |  Y  |  (X,Y)  |  Cluster   
    ---|---            ---|-----|---------|---------
    0  |  1   ----->   0  |  1  |  [0 1]  |    0
    ---|---            ---|-----|---------|---------
    2  |  3            2  |  3  |  [2 3]  |    0
This part will be helpful for calculating distances later on 
'''
dataset['(X,Y)'] = dataset.apply(lambda x: np.array((x['X'], x['Y'])), axis = 1)
dataset['Cluster'] = 0


###############################################################################
'''
Function to calculate the squared Euclidean distance between two points 
    
    Parameters: point_1 - first point
                point_2 - second point
    
    Return: Euclidean distance between point_1 and point_2

Notice that the inequality is preserved when considering the square of non-negative numbers
Eg: for distances d1, d2 >= 0 we have d1 >= d2 is equivalent to d1^2 >= d2^2
'''
def distance(point_1, point_2):
    return np.sum((point_1 - point_2)**2)


###############################################################################
'''
This part is about the K-means++ initialisation algorithm used to initialise the 
means. K-means++ is just the standard K-means but coupled with a smarter 
initialization of the centroids. The initialisation process is as below:
    1. Randomly select the first centroid from the data points.
    2. For each data point compute its distance from the nearest, previously 
    chosen centroid.
    3. Select the next centroid from the data points such that the probability 
    of choosing a point as centroid is directly proportional to its distance from 
    the nearest, previously chosen centroid. 
    (i.e. the point having maximum distance from the nearest centroid is most 
    likely to be selected next as a centroid)
    4. Repeat steps 2 and 3 until k centroids have been sampled   


Function to initialize centroids

    Parameters: dataset - given dataset
                k - desired number of clusters 
                
    Return: list of k centroids
'''
def initialize(dataset, k):
    # convert all points into numpy array for more efficient and faster distance calculation
    data_points = np.array(dataset['(X,Y)'])
    centroids = [] # list to store initial centroids

    # first pick a point at random
    centroids.append(data_points[np.random.randint(0, data_points.shape[0])])
    
    for _ in range(1,k):
        dist = [] # list to store distances of points to their nearest previously chosen centroids
        for i in range(data_points.shape[0]):
            point = data_points[i]

            # process of finding the nearest previously chosen centroid for given point  
            min_dist_with_cen = distance(point, centroids[0])
            if len(centroids) > 1:
                for j in range(1,len(centroids)):
                    dist_wrt_centroid_j = distance(point, centroids[j]) 
                    min_dist_with_cen = min(min_dist_with_cen, dist_wrt_centroid_j)

            dist.append(min_dist_with_cen) # append the mentioned distance to list "dist"

        # taking the point that has the largest respective distance to be the next centroid 
        dist = np.array(dist)
        next_centroid = data_points[np.argmax(dist)]

        # update the list of centroids
        centroids.append(next_centroid)
    return centroids


###############################################################################
'''
Function to assign a cluster to a given point based on the index of corresponding 
centroid in the given list of centroids

    Parameters: centroids - given list of centroids to assign cluster
                point - given point to be assigned cluster 
                
    Return: corresponding cluster
'''
def cluster_distribution(centroids, point):
    dist = []
    for centroid in centroids:
        dist.append(distance(point, centroid))
    return np.argmin(dist)


###############################################################################
'''
Recursive function to perform K-means clustering recursively

    Parameters: centroids - list of initial centroids
                dataset_image - a copy of given dataset  
                
    Return: clusters - clusters obtained when finish K-means clustering
            centroids - list of final centroids to calculate Sum of Squared Error later

Notice: Here we use dataset_image to preserve the format and data of the original dataset.
        In other words, we keep the original dataset as reference only. This will help prevent 
        errors came from the modification of the dataset during the recusion and later on 
'''                
def clustering(centroids, dataset_image):

    # (re)distribute all points into different clusters and accordingly keep the result in column Cluster_check 
    dataset_image['Cluster_check'] = dataset_image['(X,Y)'].apply(lambda x: cluster_distribution(centroids, x))

    # if after the centroids' coordinate-recalculation and the re-distribution of all points,
    # no point switches to another clusters, stop recursion and group the data by value of column Cluster 
    if dataset_image['Cluster_check'].equals(dataset_image['Cluster']):
        clusters = dataset_image.groupby('Cluster')
        return clusters, centroids # return the results

    # else, recursion continues
    else:

        # firstly, update the new data for column Cluster based on data of column Cluster_check 
        dataset_image['Cluster'] = dataset_image['Cluster_check']

        # re-calculation process
        clusters = dataset_image.groupby('Cluster')
        for index, cluster in clusters:

            # recalculate the coordinates of the corresponding centroid of each cluster 
            centroids[index] = np.array([cluster['X'].mean(), cluster['Y'].mean()])

        # perform recursion    
        return clustering(centroids, dataset_image)


###############################################################################
'''
Function to calculate the Sum of Squared Errors
    
    Parameters: clusters - final clusters obtained after performing K-means clustering
                centroids - list of centroids of the above clusters

    Return: the Sum of Squared Errors
'''
def sum_squared_error(clusters, centroids):
    sse = 0
    for index, cluster in clusters:
        for point in cluster['(X,Y)']:
            sse += distance(np.array(point), centroids[index])
    return sse


###############################################################################
'''
Function to assemble all the above to perform the K-means algorithm 

    Parameters: k - k value representing the desired number of clusters
                dataset - the given dataset to be clustered

    Return: final_clusters - the clustering results
            sse - Sum of Squared Errors 
'''
def run_K_means_algorithm(k, dataset):

    # initialize a list of centroids
    initial_centroids = initialize(dataset, k)

    # make a copy of given dataset as stated above
    dataset_image = dataset.copy()

    # perform k-means clustering
    final_clusters, final_centroids = clustering(initial_centroids, dataset_image)

    # calculate Sum of Squared Errors
    sse = sum_squared_error(final_clusters, final_centroids)

    # return the results
    return final_clusters, sse


###############################################################################
'''
Function to visualize each cluster

    Parameters: clusters - final clusters to visualize 
'''
def visualize_cluster(clusters, sse):
    colors = cm.gist_rainbow(np.linspace(0, 1, len(clusters)))
    for index, cluster in clusters:

        # clusters will be represented by different colors
        plt.scatter(cluster['X'], cluster['Y'], s=3, color=colors[index]) 

    plt.title('K-means Clustering with K = ' +str(len(clusters))+' (SSE = '+str(sse)+')')
    plt.xlabel('1st-attribute')
    plt.ylabel('2nd-attribute')
    plt.savefig('K_mean_'+str(len(clusters))+'.png')


###############################################################################
'''
Function to visualize the Elbow Method of finding the best k-value 

    Parameters: dataset - given dataset

Notice: The details of Elbow Method will be provided in the report
'''
def Elbow_visualize(dataset):
    list_sse = []
    for k in range(1,10):
        list_sse.append(run_K_means_algorithm(k, dataset)[1])
    plt.plot(range(1,10), list_sse, marker='o')
    plt.title('Elbow Method: WSS score against k value')
    plt.xlabel('k-value')
    plt.ylabel('sse')
    plt.savefig('Elbow_method.png')


###############################################################################
'''
Program starts running here
Notice: Please uncomment one of the group of lines below to perform (do one-by-one):
    - Clustering for K = 3
    - Clustering for K = 6
    - Clustering for K = 9
    - Elbow method
'''
#######################################
#final_clusters, sse = run_K_means_algorithm(3, dataset)
#visualize_cluster(final_clusters, sse)
#######################################
#final_clusters, sse = run_K_means_algorithm(6, dataset)
#visualize_cluster(final_clusters, sse)
#######################################
#final_clusters, sse = run_K_means_algorithm(9, dataset)
#visualize_cluster(final_clusters, sse)
#######################################
#Elbow_visualize(dataset)
#######################################

