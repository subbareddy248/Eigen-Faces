from __future__ import print_function
from PIL import Image
from numpy import *
from PIL import Image
import numpy
import pylab
import os,sys
from scipy.spatial import distance
from sklearn import svm
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings('ignore')

def tune_parameters(X, y):

	# Split the dataset in two equal parts
	X_train, X_test, y_train, y_test = train_test_split(
	    X, y, test_size=0.5, random_state=0)
	
	# Set the parameters by cross-validation
	tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
	                     'C': [1, 10, 100, 1000]},
	                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
	
	scores = ['f1']
	
	for score in scores:
	    print("# Tuning hyper-parameters for %s" % score)
	    print()
	
	    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=4,
	                       scoring='%s' % score)
	    clf.fit(X_train, y_train)
	
	    print("Best parameters set found on development set:")
	    print()
	    print(clf.best_params_)
	    print()
	    print("Grid scores on development set:")
	    print()
	    for params, mean_score, scores in clf.grid_scores_:
	        print("%0.3f (+/-%0.03f) for %r"
	              % (mean_score, scores.std() * 2, params))
	    print()
	
	    print("Detailed classification report:")
	    print()
	    print("The model is trained on the full development set.")
	    print("The scores are computed on the full evaluation set.")
	    print()
	    y_true, y_pred = y_test, clf.predict(X_test)
	    #print(classification_report(y_true, y_pred))
	    print()

	return clf.best_estimator_ 



def pca(X):
  #print X.shape
  mean=numpy.mean(X,axis=0)
  #print mean.shape
  X=X-mean
  #im = Image.new("L",(320, 243)) 
  #print mean
  #im.putdata(mean) 
  #im.save("Average_Face.png")
  cov=numpy.cov(X)
  e,ev=numpy.linalg.eigh(cov)
  es = numpy.argsort(-e)
  ev = ev[es]
  #print "ok"
  #print ev.shape,X.shape,"get i"
  ev=numpy.dot(ev,X)
  
  #print ev.shape
  return ev[:105]

path1="/home/subba/Desktop/yalefaces/"
#path2="/home/nayyar/Desktop/non-face/"
listing = numpy.array(sorted(os.listdir(path1)))
#listing1 = numpy.array(sorted(os.listdir(path2)))
#print listing
#print listing1
listing = numpy.split(listing, 15)
#print listing
ind2=0
haha=0
avg=0.
for i1 in range(0,5):
	train=[]
	test=[]
	temp=[]
	ind1=ind2
	ind2=ind2+2
	#print ind1,ind2
	for i2 in range(0,len(listing)):
		temp1=listing[i2]
		#print len(temp1)
		for ran in range(ind1,ind2):
		 test.append(listing[i2][ran])
		for ran1 in range(0,len(temp1)):
		 temp.append(temp1[ran1])
	
	#print t,"this is list"
	#for v1 in range(0,len(t))	
	test=set(test)	
	train=set(temp)
	train=train-test
	test=list(test)
	train=list(train)
	#print test,len(test),len(train)
	tag1=[]
	tag=[]
	tag3=[]
	X1=[]

	for j1 in range(0,len(train)):
	    im = Image.open((path1+str(train[j1]))).convert('L')
	    im1 = numpy.array(im).ravel()
	    X1.append(im1)
	    split=train[j1].split(".")
	    tag.append(str(split[0]))
	dtag=list(set(tag))
	#print (dtag)
	ntag=numpy.zeros(len(tag))
	
	for each in range(0,len(tag)):
		ntag[each]=dtag.index(tag[each])
	#print ntag	
	X=numpy.array(X1)
	#print tag
	V = pca(X)
	#print X.shape,V.shape
	X_train=numpy.dot(X,V.T)
	#print X_train.shape,"Note this"
	X2=[]

	for j2 in range(0,len(test)):
	    im = Image.open((path1+str(test[j2]))).convert('L')
	    im1 = numpy.array(im).ravel()
	    X2.append(im1)
	    split=test[j2].split(".")
	  
	    tag1.append(str(split[0]))
	X=numpy.array(X2)
	
	n1tag=numpy.zeros(len(tag1))
	for each in range(0,len(tag1)):
		n1tag[each]=dtag.index(tag1[each])
	
	#print tag
	
	
	#print tag1
        #print X
	#print X.shape, "this the x"
	C = 1.0
	X_test=numpy.dot(X,V.T)

	# Loading the Digits dataset
	#digits = datasets.load_digits()

	# To apply an classifier on this data, we need to flatten the image, to
	# turn the data in a (samples, feature) matrix:
	#n_samples = len(digits.images)
	#X = digits.images.reshape((n_samples, -1))
	#y = digits.target

	# Split the dataset in two equal parts 
	y_train=ntag
	y_test=n1tag

	print (y_train,y_test)
	param = tune_parameters(X_train, y_train)
	clf = SVC(C=param.C, gamma=param.gamma, kernel=param.kernel)
	clf.fit(X_train, y_train)

	y_true, y_pred = y_test, clf.predict(X_test)
	#ccc+=confusion_matrix(n1tag,y_pred)
	print(confusion_matrix(n1tag,y_pred))
	acc = len(numpy.where(y_true==y_pred)[0])/float(len(y_true))*100
	avg += acc
	print(classification_report(y_true, y_pred))
	print()
print(avg/5.)
#print(ccc/5.)
