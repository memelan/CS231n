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
