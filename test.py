import argparse

from scipy.io import loadmat

from MultiLayer import MultiLayer
from SingleLayer import SingleLayer

ap = argparse.ArgumentParser()
ap.add_argument("-test_data", "--test_data", type=str, default='./test-data.npy',
	help="test data file path")
ap.add_argument("-model_path", "--model_path", type=str, default='./model',
	help="weight model file path")
args = vars(ap.parse_args())

test = loadmat(args['test_data'])
test_x = test['x'] / 255
test_y = test['y'][0]

single_layer = SingleLayer(test_x, test_y, 1, 1, 1)
single_layer.load_model(args['model_path'])
single_layer.visualize_params()
single_layer.test(test_x, test_y)

# multi_layer = MultiLayer(test_x, test_y, 1, 1, 1)
# multi_layer.load_model(args['model_path'])
# multi_layer.test(test_x, test_y)
