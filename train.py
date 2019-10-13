import argparse

from scipy.io import loadmat

from MultiLayer import MultiLayer
from SingleLayer import SingleLayer
ap = argparse.ArgumentParser()
ap.add_argument("-data_path", "--data_path", type=str)
ap.add_argument("-e", "--epochs", type=int, default=100)
ap.add_argument("-a", "--alpha", type=float, default=0.01)
ap.add_argument("-b", "--batch_size", type=int, default=16)
args = vars(ap.parse_args())

data = loadmat(args["data_path"])
X = data['x']
y = data['y'][0]

single_layer = SingleLayer(X, y, args["epochs"], args["alpha"], args["batch_size"])
single_layer.train()
single_layer.plot_loss()

# multi_layer = MultiLayer(X, y, args["epochs"], args["alpha"], args["batch_size"])
# multi_layer.train()
# multi_layer.plot_loss()
