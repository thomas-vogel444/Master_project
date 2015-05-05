

-- Loading the MNIST dataset onto torch: See https://github.com/andresy/mnist
mnist = require "mnist"

trainingDataSet = mnist.traindataset()
testingDataSet  = mnist.testdataset()

print(trainingDataSet)
print(testingDataSet)

-- Display a bunch of images
	-- itorch notebook
	-- require "image"
	-- itorch.image(image.lena())

-- To train a Neural Network, all we need is to get a training data set, 
-- construct an architecture and train it.

-- The data set needs to be a table with an indexing [] and a size method.

-- Training a FeedForward network requires a Sequential object:
require "nn"
my_NN = nn.Sequential()

n_inputs = 3
n_outputs = 1
n_hiddenUnits = 20
my_NN:add(nn.Linear(n_inputs, n_hiddenUnits))
my_NN:add(nn.Tanh())
my_NN:add(nn.Linear(n_hiddenUnits, n_outputs))

-- We then need to train the bloody network. This requires choosing a error criterion 
-- and a learning method.
criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(my_NN, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 10
trainer.shuffleIndices = false
trainer:train(trainingDataSet)

-- We then need to test the network
