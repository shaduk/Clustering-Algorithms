import numpy as np
import pickle
import sys
import time
from sklearn.cluster import DBSCAN
import plotly
from plotly.graph_objs import *
from sklearn.decomposition import PCA as sklearnPCA

visited = []
memCluster = []

FILENAME = "new_dataset_1.txt"
EPS = 1.03
MIN_POINTS = 4

def preprocess(filename):
    inpdata = np.genfromtxt(filename,delimiter = '\t')
    X = np.loadtxt(filename,delimiter = '\t', usecols = range(2, inpdata.shape[1]), dtype = 'S15')
    gen_id = np.loadtxt(filename,delimiter = '\t', usecols = 0, dtype = 'S15')
    ground_truth = np.loadtxt(filename,delimiter = '\t', usecols = 1, dtype = 'S15')
    return X, gen_id, ground_truth

def runPCA(X):
	sklearn_pca = sklearnPCA(n_components=2)
	Y_sklearn = sklearn_pca.fit_transform(X)
	return Y_sklearn

def myDBSCAN(data, eps, minPts, distance_matrix):
	global visited
	global memCluster
	clusters = []
	noise = []
	visited = [False]*len(data)
	memCluster = [False]*len(data)
	for i in range(0, len(data)):
		if(visited[i] == False):
			visited[i] = True
			nPoints = regionQuery(i, eps, distance_matrix)
			if(len(nPoints) < minPts):
				noise.append(data[i])
			else:
				cluster = []
				cluster = expandCluster(data, i, nPoints, cluster, eps, minPts, distance_matrix)
				clusters.append(cluster)
	return clusters

def expandCluster(data, pointIndex, nPoints, cluster, eps, minPts, distance_matrix):
	global visited
	global memCluster
	cluster.append(data[pointIndex])
	memCluster[pointIndex] = True
	for i in nPoints:
		if(visited[i] == False):
			visited[i] = True
			nPoints2 = regionQuery(i, eps, distance_matrix)
			if(len(nPoints2) >= minPts):
				nPoints += nPoints2
				#nPoints.extend(nPoints2)
		if(memCluster[i] == False):
			cluster.append(data[i])
			memCluster[i] = True
	return cluster

def regionQuery(pointIndex, eps, distance_matrix):
	points = []
	for j in range(0, len(distance_matrix)):
		if(distance_matrix[pointIndex][j] <= eps):
			points.append(j)
	return points

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
	print(" Jaccard is : " + str(jaccard)),
	print(" Rand is : " + str(rand))
	

def main():
	global FILENAME
	global MIN_POINTS
	global EPS
	db = DBSCAN(eps=1.03, min_samples=4, metric="precomputed")
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
			else:
				distance_matrix[i][j] = 0
	'''
	eps_list = [1.03, 1.04]
	for n in eps_list:
		for m in range(3, 8):
			print("For eps : " + str(n)),
			print(" and for minPoints : " + str(m)),
			clusters = myDBSCAN(gen_id, n, m, distance_matrix)
			density_ground_truth = [-1]*len(gen_id)
			for i in range(0, len(clusters)):
				for j in clusters[i]:
					density_ground_truth[int(j)-1] = i
			calculateJackard(ground_truth, density_ground_truth)

	'''
	start = time.time()
	clusters = myDBSCAN(gen_id, EPS, MIN_POINTS, distance_matrix)
	print("Time is : "),
	print("--- %s seconds ---" % (time.time() - start))
	density_ground_truth = [-1]*len(gen_id)
	print("Clusters no :" + str(len(clusters)))
	print(clusters)
	for i in range(0, len(clusters)):
		for j in clusters[i]:
			density_ground_truth[int(j)-1] = i
	print(density_ground_truth)
	#y = db.fit_predict(distance_matrix)
	calculateJackard(ground_truth, density_ground_truth)
	#calculateJackard(ground_truth, y)
	Y_pca = runPCA(X)
	draw_scatter_plot(Y_pca, density_ground_truth)
	
if __name__=='__main__':
	main()