import matplotlib.pyplot as plt


def accuracy(acc):
    max_acc = [max(acc[:i+1]) for i in xrange(len(acc))]
    plt.figure(figsize=(16, 4), dpi=100)

    plt.plot(acc, color="grey", linewidth=2.5, label="Accuracy")
    plt.plot(max_acc, color="g", linewidth=2.5, label="Best accuracy")

    plt.xlabel("Iterations")
    plt.xlim(0, len(acc))

    plt.legend(loc=4)
    plt.show()
