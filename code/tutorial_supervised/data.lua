----------------------------------------------------------------------
-- This script downloads and loads the MNIST dataset
-- http://yann.lecun.com/exdb/mnist/
----------------------------------------------------------------------

train_file = 'data/mnist.t7/train_32x32.t7'
test_file  = 'data/mnist.t7/test_32x32.t7'

trainingSet = torch.load(train_file,'ascii')
testingSet  = torch.load(test_file,'ascii')

if opt.size == 'reduced' then
	-- Making a 2 class training dataset
	function get_reduced_dataset(dataset)
		local reduced_dataset_indices = {}
		for i = 1, (#dataset.labels)[1] do
			if (dataset.labels[i] == 1) or (dataset.labels[i] == 2) then
				table.insert(reduced_dataset_indices, i) 
			end
		end

		reducedDataset = {}
		reducedDataset.data   = torch.zeros(#reduced_dataset_indices, 1, 32, 32)
		reducedDataset.labels = torch.zeros(#reduced_dataset_indices)
		for i = 1, #reduced_dataset_indices do
			reducedDataset.data[{i, {}, {}, {}}] = dataset.data[reduced_dataset_indices[i]]
			reducedDataset.labels[i]			 = dataset.labels[reduced_dataset_indices[i]]
		end
		return reducedDataset
	end

	trainData = get_reduced_dataset(trainingSet)
	trainData.data = trainData.data[{{1,500}, {}, {}, {}}]
	testData  = get_reduced_dataset(testingSet)
elseif opt.size == 'small' then
	print '==> loading a small MNIST dataset'
	trainData        = {}
	trainData.data   = trainingSet.data[{{1,10000}, {}, {}, {}}]
	trainData.labels = trainingSet.labels[{{1,10000}}]
	testData 		 = testingSet
elseif opt.size == 'full' then
	print '==> loading the full MNIST dataset'
	trainData = trainingSet
	testData  = testingSet
end

trainData.size = function() return((#trainData.data)[1]) end
testData.size = function() return((#testData.data)[1]) end

trainData.data = trainData.data:float()
testData.data = testData.data:float()

