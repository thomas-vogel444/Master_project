require "utils.lua"
require "hdf5"

trainData = {}
testData = {}

-- Reading the datasets
print("Loading training data from: " .. opt.training_dataset)
local f = hdf5.open(opt.training_dataset,'r')
trainingDataset = f:read("dataset"):all():float()
trainingLabels  = f:read("labels"):all():float()

trainData.data   = trainingDataset
trainData.labels = trainingLabels
trainData.size   = function() return(trainingDataset:size()[1]) end
trainingSize     = trainData.size()
f:close()

print("Loading testing data from: " .. opt.testing_dataset)
local f = hdf5.open(opt.testing_dataset,'r')
testingDataset  = f:read("dataset"):all():float()
testingLabels   = f:read("labels"):all():float()

testData.data   = testingDataset
testData.labels = testingLabels
testData.size   = function() return(testingDataset:size()[1]) end
f:close()

-- Normalize the data
-- normalize(trainData.data)
-- normalize(testData.data)

nfeats  	= trainData.data:size()[2]
patchsize   = trainData.data:size()[3]
