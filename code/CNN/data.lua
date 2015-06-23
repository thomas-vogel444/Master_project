require "hdf5"

trainData = {}
testData = {}

-- Loading the data
filename = "../../ct_atrium/datasets/CNN_datasets.hdf5"
print("Loading data from: " .. filename)

-- Reading the datasets
local f = hdf5.open(filename,'r')
if opt.size == "small" then
	trainingDataset = f:read("training_dataset"):all():float()[{{1,500},{},{},{}}]
	testingDataset  = f:read("testing_dataset"):all():float()[{{1,500},{},{},{}}]
elseif opt.size == "full" then
	trainingDataset = f:read("training_dataset"):all():float()
	testingDataset  = f:read("testing_dataset"):all():float()
end

trainData.data = trainingDataset
trainData.size = function() return(trainingDataset:size()[1]) end

-- Creating the labels
labels = torch.ones(trainData.size())
labels[{{trainData.size()/2+1, trainData.size()}}] = 2

trainData.labels = labels

-- Reading the testing data
testData.data = testingDataset
testData.size = function() return(testingDataset:size()[1]) end

-- Creating the labels
labels = torch.ones(testData.size())
labels[{{testData.size()/2+1, testData.size()}}] = 2

testData.labels = labels

f:close()
