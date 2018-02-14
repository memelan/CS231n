# assume X_train is the data where each column is an example (e.g. 3073 x 50,000)
# assume Y_train are the labels (e.g. 10 array of 50,000)
# assume the function L evaluates the loss function

bestloss = float("inf") # Python assgins the highest possible float value
for num in xrange(1000):
  W = np.random.randn(10, 3073) * 0.0001 # generage random parameters
  loss = L(X_train, Y_train, W) # get the loss over the entire training set
  if loss <bestloss: # keep track of the best solution
    bestloss = loss
    bestW = W
  print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)

# prints:
# in attempt 0 the loss was 9.40, best 9.40
# ...


# see how well this works on test set

# assume X_test is [3073 x 10000], Y_test [10000 x 1]
scores = Wbest.dot(Xte_cols) # 10* 10000, the class scores for all the test examples
# find the index with max score in each column (the predicted class)
Yte_predict = np.argmax(scores, axis = 0)
# and calculate accuracy (fraction of predictions that are correct)
np. mean(Yte_predict == Yte)
# returns 0.1555

# SOTA state-of-the-arts ~0.95
# might not use it in practice, but is a way you could potentially do
# could work well in practice, if you got all the details right
