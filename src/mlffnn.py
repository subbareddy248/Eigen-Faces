import numpy as np
import warnings
import pylab
from PIL import Image
from PIL import Image
import os,sys
from itertools import cycle, izip
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import gen_even_slices
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
def pca(X):
  #print X.shape
  mean=np.mean(X,axis=0)
  #print mean.shape
  X=X-mean
  im = Image.new("L", (320,243)) 
  #print mean
  #im.putdata(mean) 
  im.save("Average_Face.png")
  cov=np.cov(X)
  e,ev=np.linalg.eigh(cov)
  #print "ok"
  #print ev.shape,X.shape,"get i"
  ev=np.dot(ev,X)
  
  #print ev.shape
  return ev
def _softmax(x):
    np.exp(x, x)
    x /= np.sum(x, axis=1)[:, np.newaxis]

def _tanh(x):
    np.tanh(x, x)

def _dtanh(x):
    """Derivative of tanh as a function of tanh."""
    x *= -x
    x += 1

class BaseMLP(BaseEstimator):
    """Base class for estimators base on multi layer
    perceptrons."""

    def __init__(self, n_hidden, lr, l2decay, loss, output_layer, batch_size, verbose=0):
        self.n_hidden = n_hidden
        self.lr = lr
        self.l2decay = l2decay
        self.loss = loss
        self.batch_size = batch_size
        self.verbose = verbose

        # check compatibility of loss and output layer:
        if output_layer=='softmax' and loss!='cross_entropy':
            raise ValueError('Softmax output is only supported '+
                'with cross entropy loss function.')
        if output_layer!='softmax' and loss=='cross_entropy':
            raise ValueError('Cross-entropy loss is only ' +
                    'supported with softmax output layer.')

        # set output layer and loss function
        if output_layer=='linear':
            self.output_func = id
        elif output_layer=='softmax':
            self.output_func = _softmax
        elif output_layer=='tanh':
            self.output_func = _tanh
        else:
            raise ValueError("'output_layer' must be one of "+
                    "'linear', 'softmax' or 'tanh'.")

        if not loss in ['cross_entropy', 'square', 'crammer_singer']:
            raise ValueError("'loss' must be one of " +
                    "'cross_entropy', 'square' or 'crammer_singer'.")
            self.loss = loss

    def fit(self, X, y, max_epochs, shuffle_data, verbose=0):
        # get all sizes
        n_samples, n_features = X.shape
        if y.shape[0] != n_samples:
            raise ValueError("Shapes of X and y don't fit.")
        self.n_outs = y.shape[1]
        #n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        n_batches = n_samples / self.batch_size
        if n_samples % self.batch_size != 0:
            warnings.warn("Discarding some samples: \
                sample size not divisible by chunk size.")
        n_iterations = int(max_epochs * n_batches)

        if shuffle_data:
            X, y = shuffle(X, y)

        # generate batch slices
        batch_slices = list(gen_even_slices(n_batches * self.batch_size, n_batches))

        # generate weights.
        # TODO: smart initialization
        self.weights1_ = np.random.uniform(size=(n_features, self.n_hidden))/np.sqrt(n_features)
        self.bias1_ = np.zeros(self.n_hidden)
        self.weights2_ = np.random.uniform(size=(self.n_hidden, self.n_outs))/np.sqrt(self.n_hidden)
        self.bias2_ = np.zeros(self.n_outs)

        # preallocate memory
        x_hidden = np.empty((self.batch_size, self.n_hidden))
        delta_h = np.empty((self.batch_size, self.n_hidden))
        x_output = np.empty((self.batch_size, self.n_outs))
        delta_o = np.empty((self.batch_size, self.n_outs))

        # main loop
        for i, batch_slice in izip(xrange(n_iterations), cycle(batch_slices)):
            self._forward(i, X, batch_slice, x_hidden, x_output)
            self._backward(i, X, y, batch_slice, x_hidden, x_output, delta_o, delta_h)
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        x_hidden = np.empty((n_samples, self.n_hidden))
        x_output = np.empty((n_samples, self.n_outs))
        self._forward(None, X, slice(0, n_samples), x_hidden, x_output)
        return x_output

    def _forward(self, i, X, batch_slice, x_hidden, x_output):
        """Do a forward pass through the network"""
        x_hidden[:] = np.dot(X[batch_slice], self.weights1_)
        x_hidden += self.bias1_
        np.tanh(x_hidden, x_hidden)
        x_output[:] = np.dot(x_hidden, self.weights2_)
        x_output += self.bias2_

        # apply output nonlinearity (if any)
        self.output_func(x_output)

    def _backward(self, i, X, y, batch_slice, x_hidden, x_output, delta_o, delta_h):
        """Do a backward pass through the network and update the weights"""

        # calculate derivative of output layer
        if self.loss in ['cross_entropy'] or (self.loss == 'square' and self.output_func == id):
            delta_o[:] = y[batch_slice] - x_output
        elif self.loss == 'crammer_singer':
            raise ValueError("Not implemented yet.")
            delta_o[:] = 0
            delta_o[y[batch_slice], np.ogrid[len(batch_slice)]] -= 1
            delta_o[np.argmax(x_output - np.ones((1))[y[batch_slice], np.ogrid[len(batch_slice)]], axis=1), np.ogrid[len(batch_slice)]] += 1

        elif self.loss == 'square' and self.output_func == _tanh:
            delta_o[:] = (y[batch_slice] - x_output) * _dtanh(x_output)
        else:
            raise ValueError("Unknown combination of output function and error.")

        if self.verbose > 0:
            print(np.linalg.norm(delta_o / self.batch_size))
        delta_h[:] = np.dot(delta_o, self.weights2_.T)

        # update weights
        self.weights2_ += self.lr / self.batch_size * np.dot(x_hidden.T, delta_o)
        self.bias2_ += self.lr * np.mean(delta_o, axis=0)
        self.weights1_ += self.lr / self.batch_size * np.dot(X[batch_slice].T, delta_h)
        self.bias1_ += self.lr * np.mean(delta_h, axis=0)


class MLPClassifier(BaseMLP, ClassifierMixin):
    """ Multilayer Perceptron Classifier.

    Uses a neural network with one hidden layer.
    Parameters
    ----------
    Attributes
    ----------
    Notes
    -----1
    References
    ----------"""
    def __init__(self, n_hidden=30, lr=0.1, l2decay=0, loss='cross_entropy',
            output_layer='softmax', batch_size=1, verbose=0):
        super(MLPClassifier, self).__init__(n_hidden, lr, l2decay, loss,
                output_layer, batch_size, verbose)

    def fit(self, X, y, max_epochs=500, shuffle_data=False):
        self.lb = LabelBinarizer()
        one_hot_labels = self.lb.fit_transform(y)
        super(MLPClassifier, self).fit(
                X, one_hot_labels, max_epochs,
                shuffle_data)
        return self

    def predict(self, X):
        prediction = super(MLPClassifier, self).predict(X)
        return self.lb.inverse_transform(prediction)


def test_classification():
    	path1="/home/subba/Desktop/yalefaces/"
	#path2="/home/nayyar/Desktop/non-face/"
	listing = np.array(sorted(os.listdir(path1)))
	#listing1 = np.array(sorted(os.listdir(path2)))
	#print listing
	#print listing1
	listing = np.split(listing, 15)
	#print listing
	ind2=0
	haha=0	
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
		    im1 = np.array(im).ravel()
		    X1.append(im1)
		    split=train[j1].split(".")
		    tag.append(str(split[0]))
		dtag=list(set(tag))
		ntag=np.zeros(len(tag))
	
		for each in range(0,len(tag)):
			ntag[each]=dtag.index(tag[each])
		#print ntag	
		X=np.array(X1)
		#print tag
		V = pca(X)
		#print X.shape,V.shape
		X_train=np.dot(X,V.T)
		#print X_train.shape,"Note this"
		X2=[]
		print X_train.shape
		
		for j2 in range(0,len(test)):
		    im = Image.open((path1+str(test[j2]))).convert('L')
		    im1 = np.array(im).ravel()
		    X2.append(im1)
		    split=test[j2].split(".")
		  
		    tag1.append(str(split[0]))
		X=np.array(X2)
	
		n1tag=np.zeros(len(tag1))
		for each in range(0,len(tag1)):
			n1tag[each]=dtag.index(tag1[each])
	
		#print tag
	
	
		#print tag1
		#print X
		#print X.shape, "this the x"
		C = 1.0
		X_test=np.dot(X,V.T)
		mlp = MLPClassifier()
		mlp.fit(X_train,ntag)
		k=mlp.predict(X_test)
		ccc=confusion_matrix(n1tag,k)
		print ccc
		c=0
		for each in range(0,len(tag1)):
			if(n1tag[each]==k[each]):
				c+=1
		haha=haha+ c/float(len(tag1))
		print c/float(len(tag1))
		#training_score = mlp.score(X_train, ntag)
		#print("training accuracy: %f" % training_score)
		#assert(training_score > .95)
		print(classification_report(n1tag, k))

	print "AVG===",haha/4
if __name__ == "__main__":
    test_classification()
