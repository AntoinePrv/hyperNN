from declaration import NN
from opal import ModelStructure, ModelData, Model
from opal.Solvers import NOMAD


# Return the error measure.
def get_error(parameters, measures):
    return sum(measures["acc"])


# Define parameter optimization problem.
data = ModelData(NN)
struct = ModelStructure(objective=get_error)  # Unconstrained
model = Model(modelData=data, modelStructure=struct)

# Solve parameter optimization problem.
# NOMAD.set_parameter(name='DISPLAY_STATS',
#                     value='%3dBBE  %7.1eSOL  %8.3eOBJ  %5.2fTIME')
NOMAD.solve(blackbox=model)
