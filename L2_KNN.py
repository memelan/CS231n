Class NearestNeighbor
	def _init_(self):
		pass
		
	def train(self, X, y):
		""" X is N x D where each row is an example. Y is 1-dimension of size N """
		# the nearest neighbor classifier simple remembers all the training data
		self.Xtr = X
		self.ytr = y
		
	def predict(self, X):
		""" X is N x D where each row is an example we wich to predict label for """
		num_test = X.shape[0]
		# lets meke sure that the output type matches the input type
		Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
		
		# loop over all the test rows
		for i in xrange(num_test):
			# find the nearest training image to the i'th test image
			# using the L1 distance (sum of absolute value differences)
			distances = np.sum(np.abs(self.Xtr - X[I,:]), axis = 1)
			min_index = np.argmin(distances) # get the index with smallest distance
			Ypred[i] = self.ytr[min_index] # preict the label of the nearest example
			
		return Ypred
