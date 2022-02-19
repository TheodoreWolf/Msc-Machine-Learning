
class TreeNode(object):
    """Tree class.

    (You don't need to add any methods or fields here but feel
    free to if you like. Our tests will only reference the fields
    defined in the constructor below, so be sure to set these
    correctly.)
    """

    def __init__(self, left, right, parent, cutoff_id, cutoff_val, prediction):
        self.left = left
        self.right = right
        self.parent = parent
        self.cutoff_id = cutoff_id
        self.cutoff_val = cutoff_val
        self.prediction = prediction

import numpy as np
from pylab import *
import math
from numpy.matlib import repmat
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import os
import warnings
sys.path.append('')
warnings.filterwarnings("ignore", category=DeprecationWarning)



# load in some binary test data (labels are -1, +1)
# data = loadmat("ion.data")
# xTrIon  = data['xTr'].T
# yTrIon  = data['yTr'].flatten()
# xTeIon  = data['xTe'].T
# yTeIon  = data['yTe'].flatten()
xTrIon = np.ones((281,34)) * np.arange(34)
yTrIon = np.ones((281,))
xTeIon = np.ones((70, 34)) *np.arange(34)
yTeIon = np.ones((70,))

print(xTrIon.shape, yTrIon.shape, xTeIon.shape, yTeIon.shape)


def spiraldata(N=300):
    r = np.linspace(1, 2 * np.pi, N)
    xTr1 = np.array([np.sin(2. * r) * r, np.cos(2 * r) * r]).T
    xTr2 = np.array([np.sin(2. * r + np.pi) * r, np.cos(2 * r + np.pi) * r]).T
    xTr = np.concatenate([xTr1, xTr2], axis=0)
    yTr = np.concatenate([np.ones(N), -1 * np.ones(N)])
    xTr = xTr + np.random.randn(xTr.shape[0], xTr.shape[1]) * 0.2

    xTe = xTr[::2, :]
    yTe = yTr[::2]
    xTr = xTr[1::2, :]
    yTr = yTr[1::2]

    return xTr, yTr, xTe, yTe


xTrSpiral, yTrSpiral, xTeSpiral, yTeSpiral = spiraldata(150)


def sqsplit(xTr, yTr, weights=[]):
    """Finds the best feature, cut value, and loss value.

    Input:
        xTr:     n x d matrix of data points
        yTr:     n-dimensional vector of labels
        weights: n-dimensional weight vector for data points

    Output:
        feature:  index of the best cut's feature
        cut:      cut-value of the best cut
        bestloss: loss of the best cut
    """
    N, D = xTr.shape
    assert D > 0  # must have at least one dimension
    assert N > 1  # must have at least two samples
    if weights == []:  # if no weights are passed on, assign uniform weights
        weights = np.ones(N)
    weights = weights / sum(weights)  # Weights need to sum to one (we just normalize them)
    bestloss = np.inf
    feature = np.inf
    cut = np.inf

    # YOUR CODE HERE

    # we sort to compute splits quicker
    sorted_index = np.argsort(xTr, axis=0)
    c_array = np.zeros((N - 1, D))
    loss_array = np.zeros((N - 1, D))

    # find optimal features, find optimal cut and optimal loss, pick best one
    for d in range(D):

        # we want to compute all the possible splits in dimension d
        data = np.take_along_axis(xTr[:, d], sorted_index[:, d], axis=0)
        labels = np.take_along_axis(yTr, sorted_index[:, d], axis=0)
        weights_d = np.take_along_axis(weights, sorted_index[:, d], axis=0)

        c_array[:, d] = (data[:-1] + data[1:]) / 2

        Wl = 1e-300
        Wr = np.sum(weights)
        Pl = 0
        Pr = np.sum(weights * labels)
        Ql = 0
        Qr = np.sum(weights * (labels ** 2))

        tmp_W = 0
        tmp_P = 0
        tmp_Q = 0

        for n in range(N - 1):

            if c_array[n, d] > data[n]:
                Wl += weights[n] + tmp_W
                Wr -= weights[n] - tmp_W

                Pl += weights[n] * labels[n] + tmp_P
                Pr -= weights[n] * labels[n] - tmp_P

                Ql += weights[n] * (labels[n] ** 2) + tmp_Q
                Qr -= weights[n] * (labels[n] ** 2) - tmp_Q

                tmp_W = 0
                tmp_P = 0
                tmp_Q = 0
            else:
                tmp_W += weights[n]
                tmp_P += weights[n] * labels[n]
                tmp_Q += weights[n] * (labels[n] ** 2)

            loss_array[n, d] = Ql - (Pl ** 2 / Wl) + Qr - (Pr ** 2 / Wr)

    cut_ind, feature = np.unravel_index(np.argmin(loss_array), loss_array.shape)
    bestloss = loss_array[cut_ind, feature]
    cut = c_array[cut_ind, feature]

    return feature, cut, bestloss


def cart(xTr, yTr, depth=np.inf, weights=None):
    """Builds a CART tree.

    The maximum tree depth is defined by "maxdepth" (maxdepth=2 means one split).
    Each example can be weighted with "weights".

    Args:
        xTr:      n x d matrix of data
        yTr:      n-dimensional vector
        maxdepth: maximum tree depth
        weights:  n-dimensional weight vector for data points

    Returns:
        tree: root of decision tree
    """
    n, d = xTr.shape
    if weights is None:
        w = np.ones(n) / float(n)
    else:
        w = weights

    # YOUR CODE HERE
    # root node
    # we create nodes according to either maxdepth or number of data points

    tree = builder(depth, xTr, yTr, w).carttree

    return tree


def get_lr_data(cut, feature, data, weights, labels):
    # faster: get indices
    l_indices = data[:, feature] <= cut
    r_indices = data[:, feature] > cut

    datal = data[l_indices]
    datar = data[r_indices]

    labelsl = labels[l_indices]
    labelsr = labels[r_indices]

    weightsl = weights[l_indices]
    weightsr = weights[r_indices]

    return datal, datar, labelsl, labelsr, weightsl, weightsr


class builder:
    def __init__(self, maxdepth, X, y, weights):

        self.N, self.D = X.shape
        self.maxdepth = maxdepth

        # the root node has no parent
        self.treenode_parent = "root"
        self.carttree = self.growing(X, y, weights)

    def growing(self, X, y, weights, depth=0):

        y = y.flatten()
        pred = max(set(y), key=list(y).count)

        # initialise the next node
        treenode = TreeNode(
            left=None,
            right=None,
            parent=self.treenode_parent,
            cutoff_id=None,
            cutoff_val=None,
            prediction=pred
        )

        if depth < self.maxdepth:

            # we want to find leaf nodes
            if X.shape[0] > 1:
                feature, cut, bestloss = sqsplit(X, y, weights)

                datal, datar, labelsl, labelsr, weightsl, weightsr = get_lr_data(cut, feature, X, weights, y)

                treenode.cutoff_id = feature
                treenode.cutoff_val = cut
                self.treenode_parent = treenode

                treenode.left = self.growing(datal, labelsl, weightsl, depth + 1)
                treenode.right = self.growing(datar, labelsr, weightsr, depth + 1)

        return treenode


def evaltree(root, xTe):
    """Evaluates xTe using decision tree root.

    Input:
        root: TreeNode decision tree
        xTe:  n x d matrix of data points

    Output:
        pred: n-dimensional vector of predictions
    """
    assert root is not None

    # YOUR CODE HERE

    N, D = xTe.shape
    treenode = root
    preds = []

    for data in xTe:
        assert len(data) == D
        while treenode.left is not None:
            if data[treenode.cutoff_id] <= treenode.cutoff_val:
                treenode = treenode.left
            else:
                treenode = treenode.right
        preds.append(treenode.prediction)
    return np.array(preds)

t0 = time.time()
root = cart(xTrIon, yTrIon)
t1 = time.time()

tr_err   = np.mean((evaltree(root,xTrIon) - yTrIon)**2)
te_err   = np.mean((evaltree(root,xTeIon) - yTeIon)**2)

print("elapsed time: %.2f seconds" % (t1-t0))
print("Training RMSE : %.2f" % tr_err)
print("Testing  RMSE : %.2f" % te_err)