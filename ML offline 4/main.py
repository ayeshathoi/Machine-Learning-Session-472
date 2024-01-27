import numpy as np
import matplotlib.pyplot as plt
import warnings
np.random.seed(0)
warnings.filterwarnings("ignore")
from matplotlib.patches import Ellipse
import time
import matplotlib.colors as colors
import imageio

### function to perform PCA ### TASK 1
def PCA(data,k):
    mean = np.mean(data, axis=0)
    data_centered = data - mean
    U, S, Vt = np.linalg.svd(data_centered)
    data_reduced = np.dot(data_centered, Vt.T[:,:k]) 
    for i in range(data_reduced.shape[1]):
        if data_reduced[:,i].mean() < 0:
            data_reduced[:,i] = -data_reduced[:,i]
    return data_reduced

### function to align data with k ###
def data_align_with_k(data_reduced,K):

    datapoints_class = [[] for i in range(K)]

    for i in range(len(data_reduced)):
        datapoints_class[best_class[i]].append(data_reduced[i])

    for i in range(K):
        datapoints_class[i] = np.array(datapoints_class[i])

    return datapoints_class



### function to plot the PCA data ###
def plot(data,filename):
    # plot the data
    if data.shape[1] == 2:
        plt.scatter(data[:,0],data[:,1],s=1,c='b')
        plt.savefig(filename + '_PCA_TASK1.png')
        plt.show()
    else:
        print('Data is not 2D')

### Expectation Maximization Step 1: Initialize ###
def initialize(data, k):
    clusters = []
    mu_k = np.random.uniform(-5,5.0, size=(k,data.shape[1]))
    weights = 1/k
    cov_k = np.identity(data.shape[1], dtype=np.float64)

    for i in range(k):
        clusters.append({0:weights, 1:mu_k[i], 2:cov_k})

    return clusters

### Expectation Maximization Step 2: Expectation ###
def pdf(data, mu, cov):
   
    while True:
        try:
            d = data.shape[1]
            data = data - mu # (N,d)

            if np.linalg.det(cov) == 0:
                cov += np.identity(data.shape[1], dtype=np.float64) * 1e-6
            
            cov_inverse = np.linalg.inv(cov) 
            
            numerator = np.exp(-0.5 * np.sum(np.dot(data, cov_inverse) * data, axis=1))
            denominator = np.sqrt(np.power(2 * np.pi, d) * np.linalg.det(cov))
        except:
            continue

        return numerator / denominator

### Expectation Maximization Step 2: Expectation ###
def expectation(data, clusters):
    global total_clusters_distribution

    total_clusters_distribution = np.zeros(data.shape[0], dtype=np.float64)
    for cluster in clusters:
        # cluster[3] is the responsibility of cluster k for data point n
        cluster[3] = cluster[0] * pdf(data, cluster[1], cluster[2])

    for cluster in clusters:
        # total_clusters_distribution is the sum of responsibilities of all clusters for data point n
        total_clusters_distribution += cluster[3]


    for cluster in clusters:
        cluster[3] /= total_clusters_distribution # responsibility of cluster k for data point n
        cluster[3] = np.expand_dims(cluster[3], 1) # reshaping to (N,1) for broadcasting
    return clusters

### Expectation Maximization Step 3: Maximization ###
def maximization(data, clusters):
    N = data.shape[0] # number of data points
    for cluster in clusters:
        n_k = np.sum(cluster[3]) # number of data points in cluster k
        cluster[0] = n_k / N # weight of cluster k
        cluster[1] = np.sum(cluster[3] * data, axis=0) / n_k # mean of cluster k
        cluster[2] = np.dot((cluster[3] * (data - cluster[1])).T, data - cluster[1]) / n_k # covariance of cluster k
        cluster[3] = np.expand_dims(cluster[3], 1)
    return clusters

### Expectation Maximization ALGORITHM ###
def expectation_maximization(data, k):
    epsilon = 1e-6
    while 1:
        try:
            clusters = initialize(data, k) # Step 1: Initialize
            for i in range(1000):
                clusters =maximization(data,expectation(data, clusters))

                new_log_likelihood_val = np.sum(np.log(total_clusters_distribution))
                if i > 0:
                    if new_log_likelihood_val - current_log_likelihood_value < epsilon:
                        print("Converged at iteration ",i)
                        break

                current_log_likelihood_value = new_log_likelihood_val
            break
        except:
            continue
    return clusters,current_log_likelihood_value
############################################################################################################
def colorSet():
    return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']



def countour_plot(data, k):
    epsilon = 1e-6

    while 1:
        try:
            clusters = initialize(data, k) # Step 1: Initialize
            cs = colorSet()
            gif = []
            for i in range(1000):
                clusters =maximization(data,expectation(data, clusters)) # Step 2 & 3: Expectation & Maximization
                new_log_likelihood_val = np.sum(np.log(total_clusters_distribution)) # Step 4: Compute Log Likelihood

                ####################################################################
                figure = plt.figure(figsize=(8,6))
                ax = figure.add_subplot(111)
                ax.scatter(data[:, 0], data[:, 1], c= cs[0], marker='o', s=1)

                color_index = 1

                for k in clusters:

                    U, eigenvalues, transpse_eigenvector = np.linalg.svd(k[2])
                    eigenvectors = transpse_eigenvector.T
                    eigenvalues = eigenvalues[eigenvalues.argsort()[::-1]]
                    eigenvectors = eigenvectors[:, eigenvalues.argsort()[::-1]]
                    angle = np.arctan2(eigenvectors[:, 0][0], eigenvectors[:, 0][1])

                    for j in range(1, 4):
                        center  = k[1]
                        width   = np.sqrt(eigenvalues[0]) * j * 2
                        height  = np.sqrt(eigenvalues[1]) * j * 2
                        contour = Ellipse(xy=center, width=width,height=height, angle=np.degrees(angle),fill = False)
                        contour.set_edgecolor(colors.to_rgb(cs[color_index]))
                        ax.add_artist(contour)
                    
                    color_index += 1
                    
                figure.canvas.draw()
                time.sleep(.3)
                convert_to_bytes = figure.canvas.tostring_rgb() # convert canvas to string of bytes
                # reshape canvas to 3D array of shape (width, height, 3) 
                final = np.frombuffer(convert_to_bytes, dtype='uint8').reshape(figure.canvas.get_width_height()[::-1] + (3,))
                gif.append(final)
                ####################################################################

                if i > 0:
                    if new_log_likelihood_val - current_log_likelihood_value < epsilon:
                        break

                current_log_likelihood_value = new_log_likelihood_val
                
            break
        except:
            continue
    return gif


def readData(filename):
    file = open('./Dataset/'+filename, 'r')
    line = file.readline()
    elements = line.split(',')
    data = []
    while line:
        elements = line.split(',')
        data.append(elements)
        line = file.readline()
    file.close()
    data = np.array(data).astype(np.float64)
    return data

filename = input("Enter the filename: ")
filename = filename + '.txt'
Data = readData(filename)
data_reduced = Data
if Data.shape[1] > 2:
    data_reduced = PCA(Data,2)

plot(data_reduced,filename[:-4])



### TASK 2 ###

start_k = input("Enter the starting value of k: ")
end_k = input("Enter the ending value of k: ")
start_k = int(start_k)
end_k = int(end_k)
log_likelihoods = []

best_ith_cluster = None
best_ith_ll = -np.inf
best_ith_optimal_k = None

optimal_k = None
optimal_last_ll = -np.inf
optimal_clusters = None

All_clusters = []   

for k in range(start_k,end_k+1):
    for i in range(5):
        clusters,log_likelihood_val = expectation_maximization(Data,k)
        print("k = "+str(k)+" : converged log likelihood = ", log_likelihood_val)
        if log_likelihood_val > best_ith_ll:
            best_ith_ll = log_likelihood_val
            best_ith_cluster = clusters
            best_ith_optimal_k = k

    log_likelihoods.append(best_ith_ll)
    All_clusters.append(best_ith_cluster)

    if best_ith_ll > optimal_last_ll:
        optimal_last_ll = best_ith_ll
        optimal_k = best_ith_optimal_k
        optimal_clusters = best_ith_cluster



# min_diff = np.inf
# for i in range(1,len(log_likelihoods)):
#     # convergence point is the point where the difference between two consecutive log likelihoods is minimum
#     diff = log_likelihoods[i] - log_likelihoods[i-1]
#     if diff < min_diff:
#         min_diff = diff
#         optimal_k = i + start_k - 1
#         optimal_clusters = All_clusters[i]
        
print("optimal k = ",optimal_k)

### plot log likelihood vs k ###
plt.figure(figsize=(8,6))
plt.plot(range(start_k,end_k+1), log_likelihoods)
plt.title("Log Likelihood vs K")
plt.xlabel("K")
plt.ylabel("Log Likelihood")
plt.savefig(filename[:-4] + '_log_likelihood_vs_k.png')
plt.show()

# in case you want to plot the log likelihoods vs k from seeing the plot
Wanted_k = input("Enter the value of k for which you want to plot the data: ")
Wanted_k = int(Wanted_k)
if Wanted_k >= start_k and Wanted_k <= end_k:
    optimal_k = Wanted_k
    optimal_clusters = All_clusters[Wanted_k - start_k]




### plot the data ###
responsibilities = np.zeros((optimal_k,Data.shape[0]))
for i in range(optimal_k):
    responsibilities[i] = optimal_clusters[i][3].reshape(-1)

best_class = np.argmax(responsibilities, axis=0)
clr = colorSet()
dataclass = data_align_with_k(data_reduced,optimal_k)
# dataclass[0][:,0] means all x values of class 0
# dataclass[0][:,1] means all y values of class 0

# if there is no data point in a class, then dataclass[i] will be empty
# so we will not plot that class
count = 0
for i in range(optimal_k):
    if dataclass[i].shape[0] != 0:
        plt.scatter(dataclass[i][:,0],dataclass[i][:,1],s=1,c=clr[i])
    else :
        count += 1

print("You have plotted ",optimal_k," clusters")

plt.title("GMM K = "+str(optimal_k))
plt.xlabel("x_value_of_datapoints")
plt.ylabel("y_value_of_datapoints")
plt.savefig(filename[:-4] + '_GMM_K_'+str(optimal_k)+'.png')
plt.show()

### Bonus Task ###
gif = countour_plot(data_reduced, optimal_k)
print("Converged at iteration ",len(gif))
imageio.mimsave('./'+filename[:-4]+'_GMM_K_'+str(optimal_k)+'.gif', gif, fps=10) 

