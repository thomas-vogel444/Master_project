require "hdf5"

trainData = {}
testData = {}

-- Loading the data
filename = "../../ct_atrium/datasets/CNN_datasets.hdf5"
print("Loading data from: " .. filename)

-- Reading the datasets
local f = hdf5.open(filename,'r')

trainingDataset = f:read("training_dataset"):all():float()
testingDataset  = f:read("testing_dataset"):all():float()

trainData.data = trainingDataset:div(255)
trainData.size = function() return(trainingDataset:size()[1]) end

-- Creating the labels
labels = torch.ones(trainData.size())
labels[{{trainData.size()/2+1, trainData.size()}}] = 2

trainData.labels = labels

-- Reading the testing data
labels = torch.ones(testingDataset:size()[1])
labels[{{testingDataset:size()[1]/2+1, testingDataset:size()[1]}}] = 2

if opt.size == "small" then
	n = 1000
	testData.data = testingDataset[{{(#testingDataset)[1]/2 - n/2, (#testingDataset)[1]/2 + n/2}, {}, {}, {}}]
	testData.labels = labels[{{(#testingDataset)[1]/2 - n/2, (#testingDataset)[1]/2 + n/2}}]
	testData.size = function() return(testData.data:size()[1]) end
elseif opt.size == "full" then
    testData.data = testingDataset
    testData.labels = labels
    testData.size = function() return(testingDataset:size()[1]) end
end

f:close()
