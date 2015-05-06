from PIL import Image
from numpy import *
from PIL import Image
import numpy
import pylab
import os,sys
from scipy.spatial import distance
import scipy
def pca(X):
  mean=numpy.mean(X,axis=0)
  X=X-mean
  cov=numpy.cov(X)
  e,ev=numpy.linalg.eigh(cov)
  ev=numpy.dot(ev,X)
  return ev


path1="/home/subba/Desktop/yalefaces/"
listing = numpy.array(sorted(os.listdir(path1)))
print listing
listing = numpy.split(listing, 15)
print listing
ind2=0
avacc=0
for i1 in range(0,5):
	train=[]
	test=[]
	temp=[]
	ind1=ind2
	ind2=ind2+2
	for i2 in range(0,len(listing)):
		temp1=listing[i2]
		for ran in range(ind1,ind2):
		 test.append(listing[i2][ran])
		for ran1 in range(0,len(temp1)):
		 temp.append(temp1[ran1])	
	test=set(test)	
	train=set(temp)
	
	train=train-test
	test=list(test)
	print test	
	train=list(train)
	tag1=[]
	tag=[]
	tag3=[]
	X1=[]

	for j1 in range(0,len(train)):
	    im = Image.open((path1+str(train[j1])))
	    im1 = numpy.array(im).ravel()
	    X1.append(im1)
	    tag.append(str(train[j1]))
	X=numpy.array(X1)
	print X
        V = pca(X)
	X_train=numpy.dot(X,V.T)
	X2=[]

	for j2 in range(0,len(test)):
	    im = Image.open((path1+str(test[j2])))
	    im1 = numpy.array(im).ravel()
	    X2.append(im1)
	    tag1.append(str(test[j2]))
	X=numpy.array(X2)
	X_test=numpy.dot(X,V.T)
	index=0
	tag3=[]
	for i in range(0,len(tag1)):
		min1=float('inf')
		for j in range(0,len(tag)):
			a=(X_test[i])
			b=(X_train[j])
			dst = distance.euclidean(a,b)
			
			if(dst<min1):
			 	index=j
				min1=dst
		print min1
			
		tag3.append(tag[index])
	c=0
        dist=[]
	for kk in range(0,len(tag1)):
		k11=tag1[kk].split(".")
		dist.append(k11[0])
	dist=set(dist)
	
	dist=list(dist)
 	CM=[[0 for m in range(0,len(dist))] for m1 in range(0,len(dist))]
	
	for i in range(0,len(tag1)):
			s1=tag1[i].split(".")
			s2=tag3[i].split(".")
			if(s1[0]==s2[0]):
				ind1=dist.index(s1[0])
				ind2=dist.index(s2[0])
				c+=1
				CM[ind1][ind2]+=1
	print c/float(len(tag1))
	print CM
	avacc=avacc+ c/float(len(tag1))

print "The average Accuracy::=",avacc/5


