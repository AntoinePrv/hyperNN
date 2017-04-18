import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import numpy as np


def accuracy(acc):
    max_acc = [max(acc[:i+1]) for i in xrange(len(acc))]
    plt.figure(figsize=(16, 4), dpi=100)

    plt.plot(acc, color="grey", linewidth=2.5, label="Accuracy")
    plt.plot(max_acc, color="g", linewidth=2.5, label="Best accuracy")

    plt.xlabel("Iterations")
    plt.xlim(0, len(acc))

    plt.legend(loc=4)
    plt.show()


def mds_accuracy(X, acc):
    X = MDS(n_components=2, random_state=42).fit_transform(X)
    plt.figure(figsize=(16, 4), dpi=100)
    cb = plt.scatter(X[:, 0], X[:, 1], c=acc,
                     cmap=plt.cm.get_cmap('jet'),
                     vmin=0.1, vmax=1, s=45)
    plt.colorbar(cb)
    plt.title("Accuracy in two components MDS view")
    plt.show()


def summary(acc, quantile):
    print("Best accuracy: {} at iteration {}".format(acc.max(), acc.argmax()))
    print("Number of solutions better than {0:g}%: {1:.1f}%".format(
        100 * quantile,
        100 * np.sum(acc >= quantile) / float(acc.shape[0])
    ))
