# Implementation of KMEANS
# import libraries
import numpy as np


class Kmeans:
    def __init__(self,k=8): 
        self.num_cluster = k  # Placeholders for the number of clusters
        self.center = None    # Placeholders for the postion of all k centers
        self.cluster_label = np.zeros([k])  # Placeholders for the labels of all the cluster, you don't need to create it, just access it, the shape is (k,)
        self.error_history = []  # the reconstruction error history during the training, you don't need to handle this

    # the fit function does 2 things: (1) find the optimized k-means cluster centers (2) assign label for each cluster
    # Input: 
    #   X: training data, (n, d)
    #   y: trianing_labels, (n, )
    # Output:
    #   num_iter: number of iterations, scaler
    #   error_history: a list of reconstruction errors, list
    def fit(self, X, y):
        # initialization with pre-defined centers
        dataIndex = np.array([1, 200, 500, 1000, 1001, 1500, 2000, 2005])
        self.center = initCenters(X, dataIndex[:self.num_cluster])

        # reset num_iter
        num_iter = 0

        # initialize the cluster assignment
        # cluster assignment are like follows: if there are 10 data, and 3 centers indexed by 0,1,2
        # the cluster assignment [0,0,0,0,0,1,2,1,2,2] means you assign the first 5 points to cluster 0, 
        # point 6,8 to cluster 1, rest to cluster 2
        prev_cluster_assignment = np.zeros([len(X),]).astype('int')
        cluster_assignment = np.zeros([len(X),]).astype('int')
        is_converged = False  # Flag to check whether the center stops updating

        # main-loop for kmeans update
        while not is_converged:
            # using additional space to reduce time complexity
            # For instance, assuming you have X = [1,2,3,4,5,5,5,7,8], and expect to have 2 centers, new_center['center'] = [[0], [0]] at first
            # during iteration, for each data assessed, you can add the point postion to corresponding center, if you assign x[0],x[1] to cluster 0, new_center['center'] = [[1+2], [0]]
            # and new_center['num_sample'] = [2, 0]. Until finished, you can directly compute the updated center by dividing new_center['center'] with its corresponding count new_center['num_sample']
            new_center = dict()
            new_center['center'] = np.zeros(self.center.shape)  # to save time, this variable can store the summation of point positions that are assigned to the same cluster (k,d)
            new_center['num_sample'] = np.zeros(self.num_cluster)  # this variable stores the number of points that have being assinged for each cluster (k, )

            # iterate through the samples and compute their cluster assignment (E step)
            for i in range(len(X)):
                # compute the euclidean distance between sample and centers
                distances = computeDis(X[i], self.center)
                cur_cluster = assignCen(distances)
                cluster_assignment[i] = cur_cluster
                new_center['center'][cur_cluster] += X[i]
                new_center['num_sample'][cur_cluster] += 1

            # update the centers based on cluster assignment (M step)
            self.center = updateCen(new_center)

            # compute the reconstruction error for the current iteration
            cur_error = computeError(X, cluster_assignment, self.center)
            self.error_history.append(cur_error)

            # reach convergence if the assignment does not change anymore
            is_converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1


        # compute the class label of each cluster based on majority voting
        contingency_matrix = np.zeros([self.num_cluster,3])
        label2idx = {0:0,8:1,9:2}
        idx2label = {0:0,1:8,2:9}
        for i in range(len(cluster_assignment)):
            contingency_matrix[cluster_assignment[i],label2idx[y[i]]] += 1
        cluster_label = np.argmax(contingency_matrix,-1)
        for i in range(self.num_cluster):
            self.cluster_label[i] = idx2label[cluster_label[i]]

        return num_iter, self.error_history

    def predict(self,X):
        # predicting the labels of test samples based on their clustering results
        prediction = np.ones([len(X),])  # placeholder
        # iterate through the test samples
        for i in range(len(X)):
        
            # (1) find the cluster of each sample
            # (2) get the label of the cluster as predicted label for that data

            # get distance from center
            distances = computeDis(X[i], self.center)

            # get index of closest cluster
            cluster_index = assignCen(distances)

            # get label for that closest cluster
            cluster_label = self.cluster_label[cluster_index]

            # update prediction
            prediction[i] = cluster_label


            


        return prediction

    def params(self):
        return self.center


# ---------------------------------------------------------
# You are going to implement the following helper functions
# ---------------------------------------------------------

# init K data centers specified by the dataIndex
# For example, if X = [1,2,3,4,8,10,8,8], dataIndex = [0,4,5] => centers = [1, 8, 10]
# Input: 
#   X: training data, (n, d)
#   dataIndex: list of center index, (k, )
# Output:
#   centers: (k, d)
def initCenters(X, dataIndex):
    centers = np.zeros([len(dataIndex), X.shape[1]])  # placeholders of centers, can be ignored or removed

    # assign the position of specified points as the centers
    

    # set each item in centers
    for i in range(len(dataIndex)):
        centers[i] = X[dataIndex[i]]

    return centers


# compute the euclidean distance between x to all the centers
# input:
#   x: single data, (1, d) or (d, )
#   centers:   k center position, (k, d)
# Output:
#   dis: k distances, k
def computeDis(x, centers):
    dis = np.zeros(len(centers))  # placeholders of distances to all centers, can be ignored or removed

    
    # compute distance, can use np.linalg.norm

    # for each item in centers, set distance from data point
    for i in range(len(centers)):
        dis[i] = np.linalg.norm(centers[i] - x)


    return dis


# compute the index of closest cluster for assignment
# input:
#   distances:  k distances denote the distance between x and k centers
# Output
#   centers:  1 
def assignCen(distances):
    assignment = -1  # placeholders 

    

    # get the index of min distance
    assignment = np.argmin(distances)

    return assignment


# compute center by dividing the sum of points with the corresponding count
# input:
#   new_center:  dict, structure specified in previous comments line 38-41
# Output
#   centers:   k center position, (k, d)
def updateCen(new_center):
    centers = np.zeros_like(new_center['center'])  # placeholders of centers, can be ignored or removed

   

    # get the postion of each center by division between the 
    # sum of points at one center and their corresponding count

    prev_centers = new_center['center']
    num_samples = new_center['num_sample']

    # each item in updated centers is the previous sum of vals / count of vals 
    # since the new_center['center'] has the sum of values and new_center['num_samples'] has the count
    for i in range(len(new_center['center'])):
        centers[i] = prev_centers[i]/num_samples[i]



    return centers


# compute reconstruction error, assume data X = [1,2,5,7] has assignment [0,0,1,1], and the center positions are [1.5, 6].
# the reconstruction error is (1-1.5)^2+(2-1.5)^2+(5-6)^2+(7-6)^2
def computeError(X, assign, centers):
    error = 0   # placeholders of errors, can be ignored or removed

    

    # for i in range(len(X)):
    #     for j in range(len(X[i])):
    #         error += errorHelper(X[i][j], centers[assign[i]][j])

    
    # second approach making it faster
    # above approach has 2400 for i and 784 for j so 2400 * 784 iterations....
    # use numpy functions to make it faster
            
    # first get the centers assigned for each item in X[i]
    # make assigned_centers an np array which each value is the centers array value with the index of corresponding assign val
    # for example, with the aboveassignment [0,0,1,1], and center positions [1.5, 6]
    # assigned_centers would be [1.5, 1.5, 6, 6]
    assigned_centers = centers[assign]

    # now assigned_centers is an array with the same dimensions as X
    # since each center array is the same size as each item in X and the number of things to be assigned is the number of items in X
    # so since the sizes are the same, do element wise subtraction and element wise multiplication/squaring
    matrices_diff = X - assigned_centers
    matrices_squared = matrices_diff ** 2

    # now sum all items, no need to specify row/column since everything needs to be added anyways
    error = np.sum(matrices_squared)

    return error



# basically just to do the individual calculations for (val1-val2)^2
# don't need anymore but still kept here
def errorHelper(x1, x2):
    return (x1 - x2) * (x1 - x2)
