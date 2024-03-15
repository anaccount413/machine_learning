import numpy as np


class GaussianDiscriminantBase:
    def __init__(self) -> None:
        pass

    def calculate_metrics(self, ytest, predictions):
        precision = compute_precision(ytest, predictions)
        recall = compute_recall(ytest, predictions)
        return precision, recall

class GaussianDiscriminant_C1(GaussianDiscriminantBase):
    # classifier initialization
    # input:
    #   k: number of classes (2 for this assignment)
    #   d: number of features; feature dimensions (8 for this assignment)
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))  # m1 and m2, store in 2*8 matrices
        self.S = np.zeros((k,d,d))   # S1 and S2, store in 2*(8*8) matrices
        self.p = np.zeros(2)  # p1 and p2, store in dimension 2 vectors

    # compute the parameters for both classes based on the training data
    def fit(self, Xtrain, ytrain):
        # Step 1: Split the data into two parts based on the labels
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)

        # Step 2: Compute the parameters for each class
        # m1, S1 for class1
        self.m[0,:] = computeMean(Xtrain1)
        
        self.S[0,:] = computeCov(Xtrain1)
        
        # m2, S2 for class2
        self.m[1,:]  = computeMean(Xtrain2)
        
        self.S[1,:]  = computeCov(Xtrain2)
        
        # priors for both class
        self.p = computePrior(ytrain)

    # predict the labels for test data
    # Input:
    # Xtest: n*d
    # Output:
    # Predictions: n (all entries will be either number 1 or 2 to denote the labels)
    def predict(self, Xtest):
        # placeholders to store the predictions
        # can be ignored, removed or replaced with any following implementations
        predictions = np.zeros(Xtest.shape[0])

        # 
        # Step1: plug in the test data features and compute the discriminant functions for both classes (you need to choose the correct discriminant functions)
        # you will finall get two list of discriminant values (g1,g2), both have the shape n (n is the number of Xtest)

        # formula is xT * -0.5 * inverse of class covariance then add transpose(inverse covariance * mean) * x 
        # then add -0.5 * inverse of class covariance * transpose mean * mean - 0.5 log(determinant of covariance) + log prior

        xt = np.transpose(Xtest)
        
        # 
        # Step2: 
        # if g1>g2, choose class1, otherwise choose class 2, you can convert g1 and g2 into your final predictions
        # e.g. g1 = [0.1, 0.2, 0.4, 0.3], g2 = [0.3, 0.3, 0.3, 0.4], => predictions = [2,2,1,2]

        for i in range(Xtest.shape[0]):
            # g1
            x = Xtest[i]
            xt = np.transpose(Xtest[i])
            inverse_cov_g1 = np.linalg.inv(self.S[0])
            g1_wi = -0.5 * inverse_cov_g1
            g1_w = np.transpose(inverse_cov_g1 @ self.m[0])
            g1_w0 = (-0.5 * np.transpose(self.m[0]) @ inverse_cov_g1 @ self.m[0]) - (0.5 * np.log(np.linalg.det(self.S[0]))) + np.log(self.p[0])
            g1 = (xt @ g1_wi @ x) + (g1_w @ x) + g1_w0

            #g2
            inverse_cov_g2 = np.linalg.inv(self.S[1])
            g2_wi = -0.5 * inverse_cov_g2
            g2_w = np.transpose(inverse_cov_g2 @ self.m[1])
            g2_w0 = (-0.5 * np.transpose(self.m[1]) @ inverse_cov_g2 @ self.m[1]) - (0.5 * np.log(np.linalg.det(self.S[1]))) + np.log(self.p[1])
            g2 = (xt @ g2_wi @ x) + (g2_w @ x) + g2_w0
            
            predictions[i] = 2
            if(g1 > g2):
                predictions[i] = 1
       
        return np.array(predictions)


class GaussianDiscriminant_C2(GaussianDiscriminantBase):
    # classifier initialization
    # input:
    #   k: number of classes
    #   d: number of features; feature dimensions
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))  # m1 and m2, store in 2*8 matrices
        self.shared_S =np.zeros((d,d))  # the shared convariance S that will be used for both classes
        self.p = np.zeros(2)  # p1 and p2, store in dimension 2 vectors

    # compute the parameters for both classes based on the training data
    def fit(self, Xtrain, ytrain):
        # Step 1: Split the data into two parts based on the labels
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)

        # Step 2: Compute the parameters for each class
        # m1 for class1
        self.m[0,:] = computeMean(Xtrain1)
        # m2 for class2
        self.m[1,:]  = computeMean(Xtrain2)
        # priors for both class
        self.p = computePrior(ytrain)

        # 
        # Step 3: Compute the shared covariance matrix that is used for both class
        # shared_S is computed by finding a covariance matrix of all the data 
        self.shared_S = computeCov(Xtrain)


    # predict the labels for test data
    # Input:
    # Xtest: n*d
    # Output:
    # Predictions: n (all entries will be either number 1 or 2 to denote the labels)
    def predict(self, Xtest):
        # placeholders to store the predictions
        # can be ignored, removed or replaced with any following implementations
        predictions = np.zeros(Xtest.shape[0])
        

        # 
        # Step1: plug in the test data features and compute the discriminant functions for both classes (you need to choose the correct discriminant functions)
        # you will finall get two list of discriminant values (g1,g2), both have the shape n (n is the number of Xtest)
        
        #linear discriminant
        # formula from slides
        inverse_cov = np.linalg.inv(self.shared_S)
        # g1
        g1_w = inverse_cov @ self.m[0]
        g1_w0 = -0.5 * (np.transpose(self.m[0]) @ (np.linalg.inv(self.shared_S)) @ self.m[0]) + np.log(self.p[0])

        
        # g2
        g2_w = inverse_cov @ self.m[1]
        g2_w0 = (-0.5 * (np.transpose(self.m[1]) @ (np.linalg.inv(self.shared_S)) @ self.m[1])) + np.log(self.p[1])

        # 
        # Step2: 
        # if g1>g2, choose class1, otherwise choose class 2, you can convert g1 and g2 into your final predictions
        # e.g. g1 = [0.1, 0.2, 0.4, 0.3], g2 = [0.3, 0.3, 0.3, 0.4], => predictions = [2,2,1,2]

        # check if g1 > g2, if so put 1 in predictions array, otherwise put 2

        for i in range(Xtest.shape[0]):
             g1 = (np.transpose(g1_w) @ Xtest[i]) + g1_w0
             g2 = (np.transpose(g2_w) @ Xtest[i]) + g2_w0

             predictions[i] = 2

             if(g1 > g2):
                 predictions[i] = 1


        return np.array(predictions)


class GaussianDiscriminant_C3(GaussianDiscriminantBase):
    # classifier initialization
    # input:
    #   k: number of classes (2 for this assignment)
    #   d: number of features; feature dimensions (8 for this assignment)
    def __init__(self, k=2, d=8):
        self.m = np.zeros((k,d))  # m1 and m2, store in 2*8 matrices
        self.shared_S =np.zeros((d,d))  # the shared convariance S that will be used for both classes
        self.p = np.zeros(2)  # p1 and p2, store in dimension 2 vectors

    # compute the parameters for both classes based on the training data
    def fit(self, Xtrain, ytrain):
        # Step 1: Split the data into two parts based on the labels
        Xtrain1, Xtrain2 = splitData(Xtrain, ytrain)

        # Step 2: Compute the parameters for each class
        # m1 for class1
        self.m[0,:] = computeMean(Xtrain1)
        # m2 for class2
        self.m[1,:]  = computeMean(Xtrain2)
        # priors for both class
        self.p = computePrior(ytrain)

        # 
        # Step 3: Compute the shared covariance matrix that is used for both class
        # shared_S is computed by finding a covariance matrix of all the data 

        

        self.shared_S = computeCov(Xtrain)
        
        # 
        # Step 4: Compute the diagonal of shared_S
        # [[1,2],[2,4]] => [[1,0],[0,4]], try np.diag()
        self.shared_S = np.diag(self.shared_S)

        # diag again to get 0s
        self.shared_S = np.diag(self.shared_S)


    # predict the labels for test data
    # Input:
    # Xtest: n*d
    # Output:
    # Predictions: n (all entries will be either number 1 or 2 to denote the labels)
    def predict(self, Xtest):
        # placeholders to store the predictions
        # can be ignored, removed or replaced with any following implementations
        predictions = np.zeros(Xtest.shape[0])

        
        # Step1: plug in the test data features and compute the discriminant functions for both classes (you need to choose the correct discriminant functions)
        # you will finally get two list of discriminant values (g1,g2), both have the shape n (n is the number of Xtest)
        # Please note here, currently we assume shared_S is a d*d diagonal matrix, the non-capital si^2 in the lecture formula will be the i-th entry on the diagonal

        # do the (data point - corresponding mean for feature)^2 for the entire matrix
        difference_c1 = (Xtest - self.m[0])**2
        difference_c2 = (Xtest - self.m[1])**2

        # diag to get rid of 0s
        self.shared_S = np.diag(self.shared_S)
        
        # do the divide by covariance value^2 for entire matrix
        divided_c1 = difference_c1/(self.shared_S)
        divided_c2 = difference_c2/(self.shared_S)

        # get g1 and g2, multiply the summation of divided by -0.5 and add the log of prior
        # summation basically iterates over the features, instances are kept the same
        # since features are columns and they are "merging" axis is 1 to represent adding items in the rows and columns preserved
        g1 = (-0.5 * np.sum(divided_c1, axis=1)) + np.log(self.p[0])
        g2 = (-0.5 * np.sum(divided_c2, axis=1)) + np.log(self.p[1])

        # 
        # Step2: 
        # if g1>g2, choose class1, otherwise choose class 2, you can convert g1 and g2 into your final predictions
        # e.g. g1 = [0.1, 0.2, 0.4, 0.3], g2 = [0.3, 0.3, 0.3, 0.4], => predictions = [2,2,1,2]

        # check if g1 > g2, if so put 1 in predictions array, otherwise put 2

        self.shared_S = np.diag(self.shared_S)
        predictions = np.where(g1 > g2, 1, 2)

        return np.array(predictions)


# ------------------------------------- Helper Functions start from here --------------------------------------------------------------
# Input:
# features: n*d matrix (n is the number of samples, d is the number of dimensions of the feature)
# labels: n vector
# Output:
# features1: n1*d
# features2: n2*d
# n1+n2 = n, n1 is the number of class1, n2 is the number of samples from class 2
def splitData(features, labels):
    # placeholders to store the separated features (feature1, feature2), 
    # can be ignored, removed or replaced with any following implementations
    features1 = []
    features2 = []

    # fill in the code
    # separate the features according to the corresponding labels, for example
    # if features = [[1,1],[2,2],[3,3],[4,4]] and labels = [1,1,1,2], the resulting feature1 and feature2 will be
    # feature1 = [[1,1],[2,2],[3,3]], feature2 = [[4,4]]


    for i in range(0, len(labels)):
        if(labels[i] == 1):
            features1.append(features[i])
        else:
            features2.append(features[i])



    return np.array(features1), np.array(features2)


# compute the mean of input features
# input: 
# features: n*d
# output: d
def computeMean(features):
    # placeholders to store the mean for one class
    # can be ignored, removed or replaced with any following implementations

    m = np.mean(features, axis=0)

    # fill in the code 
    # try to explore np.mean() for convenience
    return m


# compute the mean of input features
# input: 
# features: n*d
# output: d*d
def computeCov(features):
    # placeholders to store the covariance matrix for one class
    # can be ignored, removed or replaced with any following implementations
    S = np.eye(features.shape[1])

    # fill in the code
    # try to explore np.cov() for convenience
    S = np.cov(features, rowvar=False)
    return S


# compute the priors of input features
# input: 
# labels: n*1
# output: 2
def computePrior(labels):
    # placeholders to store the priors for both class
    # can be ignored, removed or replaced with any following implementations
    # p = np.array([0.5,0.5])

    # fill in the code 
    # p1 = numOf class1 / numOf all the data; same as p2

    p1 = np.count_nonzero(labels == 1)/len(labels)
    p2 = np.count_nonzero(labels == 2)/len(labels)
    p = np.array([p1, p2])
    return p


# compute the precision
# input:
# ytest: the ground truth labels of the test data, n*1
# predictions: the predicted labels of the test data, n*1
# output:
# precision: a float with size 1
def compute_precision(ytest, predictions):
    precision = 0.0 # a place holder can be neglected
    
    # fill in the code here !!!!!!!!!!!!!!!!!!!!!!
    # precision = countOf[true positive predictions] / countOf[positive predictions]
    # here we assume label==2 is the positive label

    # true positive count is the count of indices where ytest and predictions are both 2
    true_positives = np.count_nonzero((ytest == 2) & (predictions == 2))
    
    # positive prediction count is the count of indices where predictions is 2 (including false positive/should be negative)
    actual_positives = np.count_nonzero(predictions == 2)

    

    if(actual_positives != 0):
        precision = true_positives / actual_positives
    

    return precision

# compute the recall
# input:
# ytest: the ground truth labels of the test data, n*1
# predictions: the predicted labels of the test data, n*1
# output:
# recall: a float with size 1
def compute_recall(ytest, predictions):
    recall = 0.0 # a place holder can be neglected
    # true positive count is the count of indices where ytest and predictions are both 2
    true_positives = np.count_nonzero((ytest == 2) & (predictions == 2))
    
    # actual positive count is the count of indices where ytest is 2 (including false negative/should be positive)
    actual_positives = np.count_nonzero(ytest == 2)

    recall = 0.0
    

    if(actual_positives != 0):
        recall = true_positives / actual_positives

    
    return recall 