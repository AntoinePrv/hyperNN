from opal.core.algorithm import Algorithm
from opal.core.parameter import Parameter
from opal.core.measure import Measure


# Define Algorithm object.
NN = Algorithm(name="hyperNN", description="Hyperparameter optimisation")

# Register executable for NN.
NN.set_executable_command("python opal/runner.py")

# Register parameter file used by black-box solver to communicate with NN.
# NN.set_parameter_file("fd.param")
# Should be chosen automatically and hidden.

# Define parameter and register it with algorithm.
ac = Parameter(kind="categorical", default="relu",
               name="ac", description='Activation',
               neighbors={"relu":    ["tanh", "sigmoid"],
                          "tanh":    ["relu", "sigmoid"],
                          "sigmoid": ["relu", "tanh"]})
NN.add_param(ac)
lr = Parameter(kind="real", default=.01, bound=(0., 1.),
               name="lr", description="Learning rate")
NN.add_param(lr)
l1 = Parameter(kind="real", default=.01, bound=(0., 1.),
               name="l1", description="L1 regularization")
NN.add_param(l1)
l2 = Parameter(kind="real", default=.01, bound=(0., 1.),
               name="l2", description="L2 regularization")
NN.add_param(l2)
m = Parameter(kind="real", default=.01, bound=(0., 1.),
              name="m", description="Momentum")
NN.add_param(m)
d = Parameter(kind="real", default=.0001, bound=(0., .1),
              name="d", description="Decay")
NN.add_param(d)
nv = Parameter(kind="categorical", default="True",
               name="nv", description="Nesterov",
               neighbors={"True":  ["False"],
                          "False": ["True"]})
NN.add_param(nv)

# Define relevant measure and register with algorithm.
error = Measure(kind="real", name="acc", description="Accuracy of the model")
NN.add_measure(error)
