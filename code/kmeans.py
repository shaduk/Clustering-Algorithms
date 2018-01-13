import numpy as np
import collections
import pickle
import sys
import time
import plotly
from plotly.graph_objs import *
from sklearn.decomposition import PCA as sklearnPCA


FILENAME = "new_dataset_1.txt"
ITERATION_COUNT = 10
IDS = [3, 20, 9]

def preprocess(filename):
    inpdata = np.genfromtxt(filename,delimiter = '\t')
    X = np.loadtxt(filename,delimiter = '\t', usecols = range(2, inpdata.shape[1]), dtype = 'S15')
    gen_id = np.loadtxt(filename,delimiter = '\t', usecols = 0, dtype = 'S15')
    ground_truth = np.loadtxt(filename,delimiter = '\t', usecols = 1, dtype = 'S15')
    return X, gen_id, ground_truth


def kmeans(X, gen_id, iterationNo):
	global IDS
	itt = 0
	X = X.astype(np.float)
	clusters = gen_id
	centroid = []
	for m in IDS:
		centroid.append(X[m-1])
	c = 0
	while(True):
		iterationNo -= 1
		itt += 1
		c = c + 1
		new_centroid = np.empty_like(centroid)
		for i in range(0, X.shape[0]):
			clostestTo = -1
			mindist = sys.maxint
			for j in range(0, len(centroid)):
				euc_distance = euc_dis(centroid[j], X[i])
				if(euc_distance < mindist):
					mindist = euc_distance
					clostestTo = j
			clusters[i] = clostestTo
		for m in range(0, len(centroid)):
			points = []
			for i in range(0, len(clusters)):
				if(m == int(clusters[i])):
					points.append(X[i])
			points = np.array(points)
			new_centroid[m] = np.mean(points, axis = 0)
		#print(new_centroid)
		#print(collections.Counter(clusters))
		if((centroid == new_centroid).all() or iterationNo == 0):
			print("centroid : "),
			print(centroid)
			print("Iteration no : ", itt),
			return clusters
		else:
			centroid[:] = new_centroid
	return
	
def euc_dis(x, m):
	#print(x)
	#print(m)
	dis = 0
	for i in range(len(x)):
		dis = dis + np.sqrt(np.square(np.subtract(float(x[i]), float(m[i]))))	
	return dis

def runPCA(X):
	sklearn_pca = sklearnPCA(n_components=2)
	Y_sklearn = sklearn_pca.fit_transform(X)
	return Y_sklearn

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
	global K_VALUE
	global ITERATION_COUNT
	X, gen_id, ground_truth = preprocess(FILENAME)
	'''
	for i in range(2, 16):
		print("For k : " + str(i)),
		clusters = kmeans(X, gen_id, i)
		calculateJackard(ground_truth, clusters)
	'''
	start = time.time()
	clusters = kmeans(X, gen_id, ITERATION_COUNT)
	#mapreduce output clusters = ['0', '1', '0', '0', '3', '3', '3', '0', '0', '3', '0', '3', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '2', '1', '0', '1', '2', '3', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '3', '3', '0', '3', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '3', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1', '1', '1', '1', '2', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '2', '1', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '2', '1', '0', '2', '0', '0', '1', '1', '0', '0', '1', '1', '2', '1', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '2', '1', '1', '1', '1', '0', '1', '1', '2', '1', '0', '1', '1', '0', '1', '0', '0', '1', '1', '0', '0', '0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '0', '1', '1', '1', '2', '1', '2', '1', '2', '2', '1', '2', '2', '2', '1', '2', '1', '2', '2', '2', '0', '2', '0', '2', '1', '1', '1', '2', '2', '1', '2', '1', '1', '2', '1', '1', '2', '2', '2', '0', '2', '2', '0', '2', '2', '2', '1', '1', '1', '2', '2', '2', '2', '1', '1', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '1', '2', '2', '2', '1', '2', '2', '1', '1', '1', '1', '2', '1', '2', '2', '3', '2', '2', '3', '3', '2', '2', '2', '2', '2', '2', '3', '2', '0', '3', '2', '2', '3', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '2', '3', '2', '2', '2', '2', '2', '3', '3', '2', '2', '2', '3', '2', '2', '2', '2', '2', '2', '2', '0', '3', '3', '3', '3', '3', '3', '3', '3', '3', '2', '3', '3', '3', '3', '0', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '0', '3', '2', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3']
	#mapreduce output clusters = ['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '3', '3', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '3', '0', '0', '3', '0', '3', '0', '3', '3', '3', '3', '3', '1', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '3', '0', '0', '3', '3', '3', '3', '3', '0', '3', '3', '3', '3', '3', '3', '0', '0', '3', '3', '0', '0', '3', '0', '0', '3', '0', '0', '0', '0', '0', '3', '0', '0', '3', '3', '0', '3', '3', '0', '3', '3', '0', '0', '0', '0', '0', '0', '0', '0', '0', '3', '0', '3', '3', '3', '3', '0', '3', '3', '3', '2', '3', '3', '3', '3', '3', '2', '2', '0', '3', '3', '2', '3', '3', '3', '3', '0', '0', '3', '0', '3', '3', '3', '3', '3', '0', '0', '0', '0', '3', '3', '0', '3', '3', '3', '2', '2', '2', '2', '0', '3', '0', '0', '0', '0', '0', '3', '3', '0', '3', '3', '3', '3', '3', '3', '3', '1', '3', '3', '3', '0', '0', '0', '0', '0', '0', '0', '0', '3', '3', '0', '3', '3', '3', '3', '3', '3', '0', '3', '3', '3', '3', '3']
	#clusters = ['3', '1', '2', '3', '2', '2', '1', '1', '0', '0', '1', '2', '0', '3', '2', '3', '1', '0', '3', '1', '2', '0', '1', '0', '0', '3', '2', '0', '3', '0', '0', '0', '1', '0', '1', '0', '0', '0', '1', '2', '0', '0', '2', '3', '0', '2', '0', '2', '1', '3', '0', '2', '0', '0', '0', '0', '2', '1', '1', '1', '2', '1', '2', '1', '2', '0', '1', '0', '1', '0', '2', '0', '0', '0', '0', '0', '0', '0', '2', '1', '0', '0', '2', '2', '1', '3', '0', '1', '0', '3', '0', '3', '1', '0', '2', '0', '1', '1', '1', '0', '3', '0', '0', '1', '2', '0', '1', '2', '0', '0', '0', '2', '2', '0', '1', '3', '0', '3', '2', '1', '0', '2', '0', '1', '1', '1', '2', '1', '2', '2', '0', '0', '0', '2', '0', '0', '0', '0', '3', '0', '1', '3', '1', '0', '2', '0', '2', '0', '1', '1']
	print("Time to run is : "),
	print("--- %s seconds ---" % (time.time() - start))
	Y_pca = runPCA(X)
	calculateJackard(ground_truth, clusters)
	draw_scatter_plot(Y_pca, clusters)

if __name__=='__main__':
	main()