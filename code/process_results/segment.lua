require "nn"
require "torch"
require "xlua"
require "hdf5"
require "image"

cmd = torch.CmdLine()
cmd:text()
cmd:text('Segmentation')
cmd:text()
cmd:text('Options:')
cmd:option('-segmentationFile', "../../datasets/segmentation_datasets.hdf5", "Path to the segmentation file")
cmd:option('-segmentationDataset', "segmentation_dataset_fixed_z", "Segmentation dataset")
cmd:option('-predictedPath', "predicted_labels.hdf5", 'Path to the predicted file')
cmd:option('-predictedDataset', "predicted_labels_fixed_z", 'Dataset name contaiing the predicted labels')
cmd:option('-modelPath', "model.net", 'Paths to the model file')
cmd:option('-type', 'float', 'type: float | cuda')
cmd:text()
opt = cmd:parse(arg or {})

if opt.type == 'cuda' then
	require "cunn"
end

----------------------------------------------------------------------
-- 						Segment dataset
----------------------------------------------------------------------
-- Loading the segmentation dataset
print("Loading the dataset " .. opt.segmentationDataset .. " from " .. opt.segmentationFile)
local f = hdf5.open(opt.segmentationFile,'r')
data = f:read(opt.segmentationDataset):all():float()

segment_dataset = {}
segment_dataset.data = data
segment_dataset.size = function() return(segment_dataset.data:size()[1]) end

-- Normalize the data
segment_dataset.data:add(-segment_dataset.data:mean())
segment_dataset.data:div(segment_dataset.data:std())

-- Classify every voxel in the segmentation dataset
print("Segmenting the image using the model in " .. opt.modelPath)
model = torch.load(opt.modelPath):float()

prediction = torch.zeros(segment_dataset.size())
model:evaluate()

for t = 1,segment_dataset.size() do
  -- disp progress
  xlua.progress(t, segment_dataset.size())

  -- get new sample
  local input = segment_dataset.data[t]
  if opt.type == 'cuda' then
  	input:cuda()
  end

  -- test sample
  prediction[t] = model:forward(input)[2]
end

prediction = torch.round(torch.exp(prediction)) + 1

f:close()
----------------------------------------------------------------------
-- 						Save the segmentation results 
----------------------------------------------------------------------
print("Saving the segmentation results")

local f = hdf5.open(opt.predictedPath,'a')
f:write(opt.predictedDataset, prediction)
f:close()