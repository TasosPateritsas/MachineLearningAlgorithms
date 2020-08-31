from scipy.stats import multivariate_normal
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn import datasets
from sklearn.cluster import KMeans

iris = datasets.load_iris()
column_names = ["sepal_length","sepal_width","petal_length","petal_width","label"]
iris_data = iris.data
iris_target = np.reshape(iris.target,(-1,1))

concat_iris_data = np.concatenate((iris_data, iris_target), axis=1)
df = pd.DataFrame(concat_iris_data, columns = column_names)


def pick_cluster_centers(points, num_clusters = 3):

    clusters = []
    arr_idx = np.arange(len(points))
    clusters.append( (points[np.random.choice(arr_idx)],1.0 / num_clusters,np.identity(points.shape[1], dtype=np.float64)))
    init_c = KMeans(n_clusters=num_clusters, init='k-means++',random_state=0).fit(points)
    print(init_c.cluster_centers_[0])
    print(init_c.cluster_centers_[1])
    print(init_c.cluster_centers_[2])
    i=0
    while len(clusters) < num_clusters:
        clusters.append((init_c.cluster_centers_[i],1.0 / num_clusters,np.identity(points.shape[1], dtype=np.float64)))
        i+=1
    return np.array(clusters)

def e_step(points, clusters):

    def pdf_calc_func(mu, pi, Sigma):
        return lambda x: pi*stats.multivariate_normal(mu, Sigma).pdf(x)

    clust_weights = []
    for c in clusters:
        pdf = pdf_calc_func(*c)
        clust_weights.append(np.apply_along_axis(pdf, 1, points).reshape(-1,1))

    clust_weights = np.concatenate(clust_weights, axis = 1)
    # Define normalization function and normalize
    def norm_clust_weights(x):
        return [n/np.sum(x) for n in x]
    cluster_assignments = np.apply_along_axis(norm_clust_weights, 1, clust_weights)
    
    return cluster_assignments,clust_weights


def m_step(points, cluster_weights):

    new_clusts = []
    for c in cluster_weights.T:
        n_k = np.sum(c)
        pi_k = n_k / len(points) # calculate pi
        # Calculate mu
        mu_k = np.apply_along_axis(np.sum,0,points * c.reshape(-1,1)) / n_k
        # Initialize Sigma
        Sigma_k = 0
        for cw, p in zip(c, points):
            diff = p - mu_k # Find Difference
            Sigma_k += cw * np.matmul(diff.reshape(-1,1), diff.reshape(1,-1))
        # Normalize Sigma
        Sigma_k = Sigma_k / n_k
        
        new_c = (mu_k, pi_k, Sigma_k)
        new_clusts.append(new_c)
    return new_clusts

def get_log_likelihood(likelihood):
    
    log_likelihoods = np.sum(likelihood)
    return np.log(log_likelihoods)

def create_cluster_func(e_step_func, m_step_func, threshold_func, assign_args = {}):
    
    def cluster(points, centroids, max_iter = 100, stop_threshold = .0001):
        
        for i in range(max_iter):
            old_centroids = centroids
            
            cluster_weights,likelihood = e_step_func(points, centroids, **assign_args)
            
            
            log_likelihood = get_log_likelihood(likelihood)
            
            centroids = m_step_func(points, cluster_weights)
            clusters_snapshot = []
        
            for cluster in centroids:
                clusters_snapshot.append({
                    'mu_k': cluster[0],
                    'cov_k': cluster[2]
                })
            status,metric = threshold_func(centroids, old_centroids, stop_threshold)
            
            print('Iteration',i + 1,'Likelihood: ', log_likelihood)
            
            if status:
                break
        
        return (centroids,cluster_weights)
    return cluster

def basic_threshold_test(centroids, old_centroids, stop_threshold):
    
    for n, o in zip(centroids, old_centroids):

        metric = np.linalg.norm(n-o)
        if metric > stop_threshold:
            return (False,metric)
    return (True,metric)


def GMM_threshold_test(centroids, old_centroids, stop_threshold):
    for np, op in zip(centroids, old_centroids):
        status,metric = basic_threshold_test(np,op,stop_threshold)
        if not status:
            return (status,metric)
    return (status,metric)


cluster_GMM = create_cluster_func(e_step,
                                      m_step,
                                      GMM_threshold_test)

def train(df,clusters = 3,max_iter = 100):

        points = df.iloc[:,:4].values
        
        # Pick k mean initial centers
        init_cents = pick_cluster_centers(points, clusters)
        
        cents,cluster_assignments = cluster_GMM(points ,init_cents,max_iter)
        
        return (cents,cluster_assignments)

cents,cluster_assignments = train(df,3, 100)

from sklearn.mixture import GaussianMixture
from matplotlib.colors import to_hex, to_rgb
def plot_GMM(df,low_dim_df,cents,cluster_assignments,clusters = 3):

    fig, (axs) = plt.subplots(1,2, figsize = (12,6))
    

    for ax, df in zip([axs], [df]):

        points = df.iloc[:,:4].values
        
        
        #Calculate centers from sklearn
        GMM = GaussianMixture(clusters, n_init=1,covariance_type='diag').fit(points)
        cluster_assignments_sk = GMM.predict_proba(points)
        low_dim_cents_sk = update_clusters_GMM(low_dim_df.iloc[:,:2].values, cluster_assignments_sk)
        #calculate the actual centers of our GMM clustering with sklearn
        cent_sk = GMM.means_

        #Calculate centers of the low dimension data projection
        low_dim_cents_custom = update_clusters_GMM(low_dim_df.iloc[:,:2].values, cluster_assignments)


        #assign colors according to probabilities
        def find_hex(p, colors):
            p = p.reshape(-1,1)
            return to_hex(np.sum(p*colors, axis=0))
        
        colors = ['#190BF5', '#0B5A07', '#DA8DB9']#[:clusters]
        colors = [np.array(to_rgb(c)) for c in colors]
        colors = np.array(colors)
        
        #Custom Cluster
        plot_colors = [find_hex(p,colors) for p in cluster_assignments]
        
        
        # Plot each distribution in different color
        axs[0].set_title('Original')
        for cat, col in zip(low_dim_df['label'].unique(), ['#190BF5', '#0B5A07', '#DA8DB9']):
                axs[0].scatter(low_dim_df[low_dim_df.label == cat].X1, low_dim_df[low_dim_df.label == cat].X2, 
                           label = None, c = col, alpha = .45)

        axs[1].set_title('Cluster Labeled')
        axs[1].scatter(low_dim_df.X1, low_dim_df.X2, label = None, c = plot_colors, alpha = .45)
        # Plot Calculated centers
        
        only_low_dim_cents_custom = np.array([ mu for mu,pi,sigma in low_dim_cents_custom])
        only_low_dim_cents_sk = np.array([ mu for (mu,pi,sigma) in low_dim_cents_sk])
        
        axs[1].scatter(only_low_dim_cents_custom[:,0], only_low_dim_cents_custom[:,1], c = 'k', marker = 'x', label = 'Custom', s = 70)
        axs[1].scatter(only_low_dim_cents_sk[:,0], only_low_dim_cents_sk[:,1], c = 'r', marker = '+', label = 'sklearn', s = 70)
        
        # Add legend
        axs[1].legend()

plot_GMM(df,low_dim_df,cents,cluster_assignments,3)