#!/usr/bin/env python

import sys

def get_centroids(k):
	k = k - 1
	f = open("new_dataset1.txt", 'r')
	data = f.read()
	centroid = []
	for i in data.strip().split("\n"):
		row = i.strip().split("\t")
		centroid.append(row[2:])
		if(k == 0):
			return centroid
		k -= 1
		
centroids = get_centroids(4)

summ = [0]*len(centroids[0])
count = 0
cluster_points = []
last_cluster = None
current_cluster = None

open('centroid1.txt', 'w').close()

for line in sys.stdin:
	# remove leading and trailing whitespace
	line = line.strip()
	# parse the input we got from mapper.py
	current_cluster, gen_id, attributes = line.split('\t')
	attributes = attributes.split('#')

	if(last_cluster == current_cluster):
		count = count + 1
		cluster_points.append(gen_id)
		for i in range(0, len(attributes)):
			summ[i] = summ[i] + float(attributes[i])
	else:
		if(last_cluster):
			new_centroid = [x/float(count) for x in summ]
			cent = ""
			for i in new_centroid:
				cent = cent + str(float("{0:.4f}".format(i))) + "\t"
			cent = cent + "\t\n"
			with open("centroid1.txt", "a") as myfile:
				myfile.write('%s' % cent)
			print '%s\t%s\t' % (last_cluster, cluster_points)
			cluster_points = []
		summ = [0]*len(centroids[0])
		count = 1
		cluster_points.append(gen_id)
		for i in range(0, len(attributes)):
			summ[i] += float(attributes[i])
		last_cluster = current_cluster
		
if(last_cluster and last_cluster == current_cluster):
	new_centroid = [x/float(count) for x in summ]
	print '%s\t%s\t' % (last_cluster, cluster_points)
	cent = ""
	for i in new_centroid:
		cent = cent + str(float("{0:.4f}".format(i))) + "\t"
	cent = cent + "\t\n"
	with open("centroid1.txt", "a") as myfile:
		myfile.write('%s' % cent)
	cluster_points = []