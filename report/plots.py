import matplotlib.pyplot as plt
from sklearn.manifold import MDS


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
    X = MDS(n_components=2).fit_transform(X)
    plt.figure(figsize=(16, 4), dpi=100)
    cb = plt.scatter(X[:, 0], X[:, 1], c=acc,
                     cmap=plt.cm.get_cmap('jet'),
                     vmin=0.1, vmax=1, s=45)
    plt.colorbar(cb)
    plt.title("Accuracy in two components MDS view")
    plt.show()
