# -*- coding: utf-8 -*-
"""

DBSCAN_CLUSTERING
@author: PHAM Trung Kien

"""

###############################################################################
# Import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import math


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
    X  |  Y            X  |  Y  |  (X,Y)   
    ---|---            ---|-----|-------
    0  |  1   ----->   0  |  1  |  (0,1)
    ---|---            ---|-----|-------
    2  |  3            2  |  3  |  (2,3)
This part will be helpful as tuple is hashable so that each element in column (X,Y) can
be used as key in dictionary.
'''
dataset['(X,Y)'] = dataset.apply(lambda x: tuple((x['X'], x['Y'])), axis = 1)


###############################################################################
'''
This part is for calculating Euclidean distances between every two points of the dataset,
which help avoid repeatedly calculating distances when finding epsilon-neighborhood for 
all points in dataset.  

Notice: two points can be identical, which is resulted in distance = 0

dict_starts[point_1] <--- dict_ends[point_2] <--- distance between point_1 and point_2 
'''
dict_starts = {}
for point_1 in dataset['(X,Y)']:
	dict_ends = {}
	for point_2 in dataset['(X,Y)']:
		distance = math.sqrt(np.sum((np.array(point_1) - np.array(point_2))**2))
		dict_ends[point_2] = distance  
	dict_starts[point_1] = dict_ends


###############################################################################
'''
Function to decide whether a point is core point, border point or noise

	Parameters: eps - epsilon value
				min_points - min_points value

	Return: dict_neighborhood - dict_neighborhood[point_1] will
								store list of the points that have distances with
								point_1 <= epsilon
			list_core - list of core points
			list_border - list of border points
			list_noise - list of noise  
'''
def get_info(eps, min_points):
	dict_neighborhood = {}
	for point in dataset['(X,Y)']:
		dict_neighborhood[point] = [end for end, distance in dict_starts[point].items() if distance <= eps]

	# a point p is considered as core point if dict_neighborhood[p] has >= min_points elements 
	list_core   = [point for point in dataset['(X,Y)'] if len(dict_neighborhood[point]) >= min_points]

	# pre_list_border contains all points that are not core points
	pre_list_border = [point for point in dataset['(X,Y)'] if point not in list_core]

	list_border, list_noise = [], []

	# among points that are not core points, a point p is considered as border point if dict_neighborhood[p]
	# contains at least one core point
	for point in pre_list_border:
		for end in dict_neighborhood[point]:
			if end in list_core:
				list_border.append(point)
				break

	# points that are neither core nor border points will be considered as noise
	list_noise = [point for point in dataset['(X,Y)'] if point not in list_core and point not in list_border] 

	return dict_neighborhood, list_core, list_border, list_noise 


###############################################################################
'''
This part is simply for creating a dictionary with keys are all the points and there status 
of having been assigned to a cluster or not

In details, dict_unassigned[p] = 1 if point p has not been assigned to any cluster and
dict_unassigned[p] = 0 otherwise
'''
dict_unassigned = {}
for point in dataset['(X,Y)']:
	dict_unassigned[point] = 1 # initialize the status by 1 for every points


###############################################################################
'''
Recursion function to form a cluster of a given core point

	Parameters: core - given core point to consider
				dict_neighborhood - neighborhood dictionary
				list_border - list of all border points
				cluster - a list to store the results

	Return: cluster of given core point  
'''
def cluster_of_core(core, dict_neighborhood, list_border, cluster):

	# consider the neighborhood of given core point 
	for point in dict_neighborhood[core]:

		# consider only points which have not been assigned to any cluster 
		if dict_unassigned[point] == 1:

			# if point has not been appended to the current cluster, append it and set the
			# status to 0, representing it has been assigned to the a cluster
			if point not in cluster:
				cluster.append(point)
				dict_unassigned[point] = 0

				# if point is not a border point, continue recursion 
				if point not in list_border:
					cluster_of_core(point, dict_neighborhood, list_border, cluster)
	return cluster


###############################################################################
'''
Function to assemble all the above to perform DBSCAN algorithm

	Parameters: eps - given epsilon value
				min_points - given min_points value

	Return: list_cluster - list of all clusters obtained
			list_noise - list of all noise

Notice: In DBScan, there may happen to have some border points which are density-reachable
from many core points but those core points belong to different clusters. To deal with this
situation, those border points will be distributed to the cluster in the first-come-first-serve 
basis. 
	Eg. Let's take border point B and core points C1 and C2 to face the mentioned situation. 
		If C1 is executed (considered) before C2 in the program, then B belongs to cluster of C1
		In contrast, B will belong to cluster C2.   
'''
def run_DBSCAN(eps, min_points):
	list_cluster = []

	# retrieve necessary information: dict_neighborhood, list_core, list_border, list_noise
	dict_neighborhood, list_core, list_border, list_noise = get_info(eps, min_points)

	# consider all core points
	for point in list_core:

		# consider only those core points that have not been assigned to any cluster
		if dict_unassigned[point] == 1: 
			new_cluster = cluster_of_core(point, dict_neighborhood, list_border, [])
			list_cluster.append(new_cluster)

	return list_cluster, list_noise


###############################################################################
'''
Function to visualize each cluster

    Parameters: clusters - final clusters to visualize
    			noise - final noise to visualize
    			eps - given epsilon
    			min_points - given min_points
'''
def visualize(clusters, noise, eps, min_points):

	# sort clusters by length of each cluster for better visualization (better color chosen for each cluster)
    clusters.sort(key=len, reverse=True) 

    '''
    Please uncomment one of the group of lines below to visualize (do one-by-one):
   		- DBSCAN for eps = 1, min_points = 4
   		- DBSCAN for eps = 5, min_points = 4
   		- DBSCAN for eps = 5, min_points = 10
	'''

    #######################################################################
    #colors_for_cluster = cm.gist_rainbow(np.linspace(0, 1, len(clusters)))
    #
    #for cluster, color in zip(clusters, colors_for_cluster):
    #    cluster = np.array(cluster)
    #    plt.scatter(cluster[:, 0], cluster[:, 1], s = 5, color = color)
    #
    #noise = np.array(noise) 
    #
    #plt.scatter(noise[:, 0], noise[:, 1], s = 1, color = 'black')
    #######################################################################
    #colors_for_large_cluster = cm.gist_rainbow(np.linspace(0, 0.3, 6))
    #
    #for cluster, color in zip(clusters[:6], colors_for_large_cluster):
    #    cluster = np.array(cluster)
    #    plt.scatter(cluster[:, 0], cluster[:, 1], s = 3, color = color)
    #
    #colors_for_small_cluster = cm.gist_rainbow(np.linspace(0.301, 1, len(clusters)-6))
    #
    #for cluster, color in zip(clusters[6:], colors_for_small_cluster):
    #    cluster = np.array(cluster)
    #    plt.scatter(cluster[:, 0], cluster[:, 1], s = 3, color = color)
    #
    #noise = np.array(noise) 
    #
    #plt.scatter(noise[:, 0], noise[:, 1], s = 3, color = 'black')
    #######################################################################
    #colors_for_cluster = cm.gist_rainbow(np.linspace(0, 1, len(clusters)))
    #
    #for cluster, color in zip(clusters, colors_for_cluster):
    #    cluster = np.array(cluster)
    #    plt.scatter(cluster[:, 0], cluster[:, 1], s = 3, color = color)
    #
    #noise = np.array(noise)
    #
    #plt.scatter(noise[:, 0], noise[:, 1], s = 3, color = 'black')
    #######################################################################

    plt.title('DBSCAN Clustering with eps = '+str(eps)+ ', min_points = '+str(min_points))
    plt.xlabel('1st-attribute')
    plt.ylabel('2nd-attribute')
    plt.savefig('DBSCAN_result_eps_'+str(eps)+'_minpoints_'+str(min_points)+'.png')


###############################################################################
'''
Program starts running here
Notice: Please uncomment one of the group of lines below to perform 
		(do one-by-one accordingly to the group of lines uncommented in the visualize function):
   	- DBSCAN for eps = 1, min_points = 4
   	- DBSCAN for eps = 5, min_points = 4
   	- DBSCAN for eps = 5, min_points = 10
'''
##################################################
#final_clusters, final_noise = run_DBSCAN(1, 4)
#visualize(final_clusters, final_noise, 1, 4)
##################################################
#final_clusters, final_noise = run_DBSCAN(5, 4)
#visualize(final_clusters, final_noise, 5, 4)
##################################################
#final_clusters, final_noise = run_DBSCAN(5, 10)
#visualize(final_clusters, final_noise, 5, 10)
##################################################