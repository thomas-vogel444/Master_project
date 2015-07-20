require "utils.lua"
require "hdf5"

trainData = {}
testData = {}

-- Loading the data
filename = "../../datasets/CNN_datasets.hdf5"
print("Loading data from: " .. filename)

-- Reading the datasets
local f = hdf5.open(filename,'r')

trainingDataset = f:read("training_dataset"):all():float()
trainingLabels  = f:read("training_labels"):all():float()
testingDataset  = f:read("testing_dataset"):all():float()
testingLabels   = f:read("testing_labels"):all():float()

trainData.data   = trainingDataset:div(255)
trainData.labels = trainingLabels
trainData.size   = function() return(trainingDataset:size()[1]) end

testData.data   = testingDataset:div(255)
testData.labels = testingLabels
testData.size   = function() return(testingDataset:size()[1]) end

f:close()

-- Normalize the data
normalize(trainData.data)
normalize(testData.data)