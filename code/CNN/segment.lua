require "torch"
require "xlua"
require "hdf5"

filename = "../../ct_atrium/datasets/CNN_datasets.hdf5"
print("Loading segmentation data from: " .. filename)

-- Loading the segmentation dataset
local f = hdf5.open(filename,'r')
data = f:read("segmentation_dataset"):all():float()

segmentDataset = {}
segmentDataset.data = data
segmentDataset.size = function() return(segmentDataset.data:size()[1]) end

-- Classify every voxel in the segmentation dataset
prediction = torch.zeros(segmentDataset.size(),2)
model:evaluate()

for t = 1,segmentDataset.size() do
  -- disp progress
  xlua.progress(t, segmentDataset.size())

  -- get new sample
  local input = segmentDataset.data[t]
  if opt.type == 'cuda' then input = input:cuda() end

  -- test sample
  prediction[{{t}, {}}] = model:forward(input)
end

-- Save the classification into a hdf5 file for later processing
local segmentationFile = hdf5.open('results/segmentation_prediction.hdf5', 'w')
segmentationFile:write("segmentation", prediction)
segmentationFile:close()