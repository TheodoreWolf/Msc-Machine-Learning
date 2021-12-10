from matplotlib import pyplot as plt
import numpy as np
import math
import random
import pandas as pd
def random_():
  n = 4
  xdata = [1,2,3,4]
  ydata = [3,2,0,5]
  x = np.array([xdata])
  X = x**0
  for order in range(1,n):
    X = np.append(X, x**order, axis=0)
    print("xorder",x**order)
    print("matrix",X)

class polynomial_regression:
  def __init__(self, k, features, labels):
    self.k = k
    self.features = features
    self.labels = labels
    self.design = np.array([])
    self.w = np.array([])
    self.err = 0

  # run this method to train
  def train(self):

    # make design matrix
    self.design = np.array([self.features ** i for i in range(self.k)]).T

    # find the weights
    self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.design.T, self.design)), self.design.T), self.labels)

  # define the output polynomial function for plotting
  def function(self, x):

    y = 0
    for i in range(self.k):
      y += self.w[i] * x ** i
    return y

  # define the mean-squared error function
  def error(self):
    for i in range(len(self.labels)):
      self.err += (self.labels[i] - self.function(self.features[i])) ** 2
    self.err = self.err / len(self.labels)
    return self.err

# part c
def part_c():
  f = lambda x: np.sin(2*np.pi*x)**2
  x = np.linspace(0,1,100)
  # Plotting 1000 random data points and 1000 random error values
  np.random.seed(1)
  testdata = np.random.random_sample((1000,1))
  error = np.random.normal(0, 0.07, 1000)
  xtest = []
  ytest = []

  # Running each point through our function
  for n, datapoint in enumerate(testdata):
    g = f(datapoint) + error[n]
    xtest.append(float(datapoint))
    ytest.append(float(g))
    plt.plot(datapoint, g, ".")

  # not the most effective to rerun the loop but its quite fast so is ok
  orders = [2,5,10,14,18]
  MSEdata = []
  for order in orders:
    # We find the weights for each order polynomial requested
     polyfit = polynomial_regression(order, np.array(xdataset), np.array(ydataset))
     polyfit.train()
     polyfit.function(x)
     MSE = polyfit.error_to_test(xtest, ytest)
     MSEdata.append((order, MSE))
  x, y = zip(*MSEdata)
  plt.plot(x,np.log(y))

data = pd.read_csv("http://www0.cs.ucl.ac.uk/staff/M.Herbster/boston-filter/Boston-filtered.csv")
training_points = int(round(506*2/3, 0))
training_data = pd.DataFrame()
test_data = data
row_numbers = random.sample(range(0, 506), training_points)
rows = test_data.iloc[row_numbers]
training_data = training_data.append(rows)
test_data = test_data.drop(index=row_numbers)
training_data = training_data.sort_index()
print(training_data, test_data)

