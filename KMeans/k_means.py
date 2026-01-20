import numpy as np
import matplotlib.pyplot as plt


def plot_progress_kMeans(X, centroidHistory, clusters, K, maxIter):
    #Plot the result
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(X[:,0],X[:,1],c=clusters,cmap='rainbow',label='data')
    
    #Plot the centroids as black x's
    ax.scatter(centroidHistory[0:K*maxIter,0], centroidHistory[0:K*maxIter,1],marker='x',c='k')
    
    #Plotting progress line
    for i in range(0,K):
        ax.plot(centroidHistory[i:K*maxIter:3,0], centroidHistory[i:K*maxIter:3,1],c='k')

        
    #Title for the plot
    plt.title('Iteration number %d'% maxIter)
    plt.xlim(-2,10)
    plt.ylim(-2,8)
    plt.show()

def kMeans_init_centroids(X, K):
    """ This function initializes K centroids that are to be used in K-Means on the dataset X
    returns K initial centroids to be used with the K-Means on the dataset X"""
    m,n = X.shape
    centroids = np.zeros((K, n));

    # Randomly reorder the indices of examples
    randidx = np.random.permutation(m)

    # Take the first K examples as centroids
    centroids = X[randidx[0:K], :]

    return centroids

def find_closest_centroids(X, centroids):
    #Finding sample size
    m = np.shape(X)[0]
    
    #Getting cluster size
    K = np.shape(centroids)[0]
    
    #Initializing cluster array
    clusters = np.zeros((m, 1))

    for i in range(0,m):
        #Initializing belonging cluster as 1
        clusterIdx = 0
        
        #Initializing the min-distance with the distance between x(i) and first-centroid
        minDistance = np.sum(np.square(X[i,:] - centroids[0,:]))
        
        #Iterating over each centroid to find the min-distance
        for j in range(1,K):
            distance = np.sum(np.square(X[i,:] - centroids[j,:]))
            if(distance < minDistance):
                minDistance = distance
                clusterIdx = j
                
        #Assigning the cluster value to a test-item       
        clusters[i] = clusterIdx

    return clusters

def compute_centroids(X, clusters, K):
    #Finding sample size
    m = np.shape(X)[0]
    
    #Finding number of variables
    n = np.shape(X)[1]
    
    #Initializing centroids array
    centroids = np.zeros((K, n))

    for clusterNum in range(0,K):
        #Getting elements belonging to perticular cluster
        clusterEles = X[(np.ravel(clusters) == clusterNum)]
        
        #Getting no. of elements in the cluster
        nCluster = np.shape(clusterEles)[0]
        
        #Calculating mean of cluster elements to find clusterCentoid
        # centroids[clusterNum, :] =  [(1/nCluster)*np.sum(clusterEles[:,0]),(1/nCluster)*np.sum(clusterEles[:,1])]
        centroids[clusterNum, :] =  np.mean(clusterEles,axis=0)
        
        
    return centroids

def kmeans(X, initialCentroids, maxIters,plot_progress=False):
    
    #Finding sample size and number of features
    m,n = np.shape(X)
    K = np.shape(initialCentroids)[0]
    
    clusters = np.zeros((m, 1))
    
    #Initializing centroidal history array
    centroidHistory = np.zeros((K*maxIters,n))
    centroidHistory[0:K,:] = initialCentroids
    centroids = np.copy(initialCentroids)

    #Running K-Means
    for i in range(1,maxIters):

        #Output progress
        print('K-Means iteration %d/%d...\n' % (i, maxIters))
        
        #For each example in X, assign it to the closest centroid
        clusters = find_closest_centroids(X, centroids)
        # print("clusters :",clusters)

        #Optionally, plot progress here
        if plot_progress:
            plot_progress_kMeans(X, centroidHistory, clusters, K, i)

        #Given the memberships, compute new centroids
        centroids[:,:] = centroidHistory[K*i:K*(i+1),:] = compute_centroids(X, clusters, K)
        
    return centroids,clusters
