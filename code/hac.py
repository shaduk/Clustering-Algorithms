import numpy as np
import pickle
import time
import sys
import plotly
from plotly.graph_objs import *
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.cluster import AgglomerativeClustering

FILENAME = "new_dataset_2.txt"
K_VALUE = 3


def preprocess(filename):
    inpdata = np.genfromtxt(filename,delimiter = '\t')
    X = np.loadtxt(filename,delimiter = '\t', usecols = range(2, inpdata.shape[1]), dtype = 'S15')
    gen_id = np.loadtxt(filename,delimiter = '\t', usecols = 0, dtype = 'S15')
    ground_truth = np.loadtxt(filename,delimiter = '\t', usecols = 1, dtype = 'S15')
    return X, gen_id, ground_truth

def agglomerative_clus(gen_id, distance_matrix, k):
	clusters = gen_id.tolist()
	cluster_pca = []
	for i in gen_id:
		cluster_pca.append([i])
	while(len(clusters) > k):
		in1, in2 = find_min_indices(distance_matrix)
		left = -1
		right = -1
		if(in1 < in2):
			left = in1
			right = in2
		else:
			right = in1 
			left = in2
		merged_cluster = '('+clusters[left] + ',' + clusters[right] + ')'
		clusters[left] = merged_cluster
		cluster_pca[left] = cluster_pca[left] + cluster_pca[right]
		del clusters[right]
		del cluster_pca[right]
		for i in range(0, len(distance_matrix)):
			distance_matrix[left][i] = min(distance_matrix[left][i], distance_matrix[right][i])
			distance_matrix[i][left] = min(distance_matrix[left][i], distance_matrix[right][i])
		distance_matrix = np.delete(distance_matrix, right, 0)
		distance_matrix = np.delete(distance_matrix, right, 1)
	return cluster_pca, clusters
	
def draw_scatter_plot(Y, labels):
    unique_labels = set(labels)
    points = []
    for name in unique_labels:
        x = []
        y = []
        for i in range(0, len(labels)):
            if(labels[i] == name):
                x.append(Y[i,0])
                y.append(Y[i,1])
        x = np.array(x)
        y = np.array(y)
        point = Scatter(
            x = x,
            y = y,
            mode='markers',
            name=name,
            marker=Marker(size=12, line=Line(color='rgba(217, 154, 217, 123)',width=0.5),opacity=0.9))
        points.append(point)
    data = Data(points)
    layout = Layout(xaxis=XAxis(title='PC1', showline=True),
                    yaxis=YAxis(title='PC2', showline=True))
    fig = Figure(data=data, layout=layout)
    plotly.offline.plot(fig)

def find_min_indices(distance_matrix):
	ind1 = -1
	ind2 = -1
	min_elem = sys.maxint
	for i in range(0, distance_matrix.shape[0]):
		for j in range(0, distance_matrix.shape[1]):
			if(i != j):
				if(distance_matrix[i][j] < min_elem):
					min_elem = distance_matrix[i][j]
					ind1 = i 
					ind2 = j
	return ind1, ind2


def inbuiltH(X, n_clusters, gen_id):
	model = AgglomerativeClustering(n_clusters=n_clusters,linkage="average", affinity="euclidean")
	print("Print inbuilt HClustering: ")
	ans = model.fit_predict(X, y = gen_id)
	print(ans)
	print(len(ans))
	return ans


def calculateJackard(ground_truth, heirarical_ground_truth):
	m00 = 0
	m01 = 0
	m10 = 0
	m11 = 0
	for i in range(0, len(ground_truth)):
		for j in range(0, len(ground_truth)):
			if((ground_truth[i] != ground_truth[j]) and (heirarical_ground_truth[i] != heirarical_ground_truth[j])):
				m00 += 1
			elif((ground_truth[i] == ground_truth[j]) and (heirarical_ground_truth[i] != heirarical_ground_truth[j])):
				m01 += 1
			elif((ground_truth[i] != ground_truth[j]) and (heirarical_ground_truth[i] == heirarical_ground_truth[j])):
				m10 += 1
			elif((ground_truth[i] == ground_truth[j]) and (heirarical_ground_truth[i] == heirarical_ground_truth[j])):
				m11 += 1
	jaccard = m11 / float(m11 + m10 + m01)
	rand = (m11 + m00) / float(m11 + m10 + m01 + m00)
	print(" Jacc is : " + str(jaccard)),
	print(" Rand is : " + str(rand))

def runPCA(X):
	sklearn_pca = sklearnPCA(n_components=2)
	Y_sklearn = sklearn_pca.fit_transform(X)
	return Y_sklearn

def main():
	global FILENAME
	global K_VALUE
	X, gen_id, ground_truth = preprocess(FILENAME)
	len_X = X.shape[0]
	distance_matrix = np.zeros((len_X, len_X), dtype='float64')
	print(X.shape)
	
	for i in range(0, len_X):
		for j in range(0, len_X):
			if(i != j):
				dis = 0
				for k in range(0, X.shape[1]):
					dis = dis + np.square(np.subtract(float(X[i][k]), float(X[j][k])))
				distance_matrix[i][j] = np.sqrt(dis) 
				distance_matrix[j][i] = np.sqrt(dis)
	'''
	for k in range(1, 10):
		clusters_pca = agglomerative_clus(gen_id, distance_matrix, k)
		#print("My algo Clusters : ")
		#print(clusters_pca)
		heirarical_ground_truth = [0]*len(gen_id)
		for i in range(0, len(clusters_pca)):
			for j in clusters_pca[i]:
				heirarical_ground_truth[int(j)-1] = i
		print("For k : " + str(k)),
		calculateJackard(ground_truth, heirarical_ground_truth)
	'''
	#heirarical_ground_truth = inbuiltH(X, 10, gen_id)
	start = time.time()
	clusters_pca, clusters = agglomerative_clus(gen_id, distance_matrix, K_VALUE)
	print(clusters)
	heirarical_ground_truth = [0]*len(gen_id)
	for i in range(0, len(clusters_pca)):
		for j in clusters_pca[i]:
			heirarical_ground_truth[int(j)-1] = i
	print("Time is : "),
	print("--- %s seconds ---" % (time.time() - start))
	calculateJackard(ground_truth, heirarical_ground_truth)
	Y_pca = runPCA(X)
	draw_scatter_plot(Y_pca, heirarical_ground_truth)
	'''
	USACities = ['BOS','NY','DC','MIA','CHI','SEA','SF','LA','DEN']
	USADistances = [
		[   0,  206,  429, 1504,  963, 2976, 3095, 2979, 1949],
		[ 206,    0,  233, 1308,  802, 2815, 2934, 2786, 1771],
		[ 429,  233,    0, 1075,  671, 2684, 2799, 2631, 1616],
		[1504, 1308, 1075,    0, 1329, 3273, 3053, 2687, 2037],
		[ 963,  802,  671, 1329,    0, 2013, 2142, 2054,  996],
		[2976, 2815, 2684, 3273, 2013,    0,  808, 1131, 1307],
		[3095, 2934, 2799, 3053, 2142,  808,    0,  379, 1235],
		[2979, 2786, 2631, 2687, 2054, 1131,  379,    0, 1059],
		[1949, 1771, 1616, 2037,  996, 1307, 1235, 1059,    0]]

	distance_matrix = np.array(USADistances)
	print(find_min_indices(distance_matrix))
	agglomerative_clus(USACities, distance_matrix, 2)
	'''
	'''
	ItalyCities = ['BA','FI','MI','NA','RM','TO']
	ItalyDistances = [
		[  0, 662, 877, 255, 412, 996],
		[662,   0, 295, 468, 268, 400],
		[877, 295,   0, 754, 564, 138],
		[255, 468, 754,   0, 219, 869],
		[412, 268, 564, 219,   0, 669],
		[996, 400, 138, 869, 669,   0]]
	distance_matrix = np.array(ItalyDistances)
	print(find_min_indices(distance_matrix))
	agglomerative_clus(ItalyCities, distance_matrix, 1)
	'''
	'''
	ItalyCities = ['P1','P2','P3','P4','P5','P6']
	ItalyDistances = [
		[  0, 0.23, 0.22, 0.37, 0.34, 0.23],
		[0.23,   0, 0.15, 0.20, 0.14, 0.25],
		[0.22, 0.15,   0, 0.15, 0.28, 0.11],
		[0.37, 0.20, 0.15,   0, 0.29, 0.22],
		[0.34, 0.14, 0.28, 0.29,   0, 0.39],
		[0.23, 0.25, 0.11, 0.22, 0.39,   0]]
	distance_matrix = np.array(ItalyDistances)
	clusters_pca = agglomerative_clus(ItalyCities, distance_matrix, 2)
	heirarical_ground_truth =  {}
	for i in range(0, len(clusters_pca)):
		for j in clusters_pca[i]:
			heirarical_ground_truth[j] = i
	print(heirarical_ground_truth)
	'''
if __name__=='__main__':
	main()



