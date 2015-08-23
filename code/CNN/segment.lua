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
cmd:option('-GPU_id', 1, "Which GPU to use")
cmd:option('-number_of_GPUs', 1, "Which GPU to use")
cmd:option('-segmentationPath', "../../datasets/segmentation_datasets.hdf5", "Path to the segmentation file")
cmd:option('-segmentationDataset', "dataset", "Segmentation dataset")
cmd:option('-height', 480, "height of the segmented image")
cmd:option('-width', 480, "width of the segmented image")
cmd:option('-predictedPath', "predicted_labels.hdf5", 'Path to the predicted file')
cmd:option('-predictedDataset', "dataset", 'Dataset name containing the predicted labels')
cmd:option('-modelPath', "model.net", 'Paths to the model file')
cmd:text()
opt = cmd:parse(arg or {})

require "cunn"
require "cudnn"
require "fbcunn"
require "fbnn"
cutorch.setDevice(opt.GPU_id)

----------------------------------------------------------------------
-- 						     Load and normalizing the dataset
----------------------------------------------------------------------
-- Loading the segmentation dataset
print("Loading the dataset " .. opt.segmentationDataset .. " from " .. opt.segmentationPath)
local f = hdf5.open(opt.segmentationPath,'r')
data    = f:read(opt.segmentationDataset):all():float()
f:close()

segment_dataset = {}
segment_dataset.data = data
segment_dataset.size = function() return(segment_dataset.data:size()[1]) end

-- Normalize the data
-- segment_dataset.data:add(-segment_dataset.data:mean())
-- segment_dataset.data:div(segment_dataset.data:std())

----------------------------------------------------------------------
--            Segment the dataset
----------------------------------------------------------------------
-- Copying the model onto all the GPUs required
model = torch.load(opt.modelPath)

-- Multi-GPU set up
if opt.number_of_GPUs > 1 then
    print('Using data parallel')
    local GPU_network = nn.DataParallel(1):cuda()
    for i = 1, opt.number_of_GPUs do
        local current_GPU = math.fmod(opt.GPU_id + (i-1)-1, cutorch.getDeviceCount())+1
        cutorch.setDevice(current_GPU)
        GPU_network:add(model:clone():cuda(), current_GPU)
    end
    cutorch.setDevice(opt.GPU_id)

    model = GPU_network
end

-- Classify every voxel in the segmentation dataset
print("Segmenting the image using the model in " .. opt.modelPath)

prediction = torch.zeros(segment_dataset.size())
model:evaluate()

-- Transfer to the GPU
model:cuda()
prediction = prediction:cuda()

local batchSize = 1500*opt.number_of_GPUs

-- Segment the whole dataset
for t = 1,segment_dataset.size(),batchSize do
  	-- disp progress
  	xlua.progress(math.min(t + batchSize -1, segment_dataset.size()), segment_dataset.size())

    -- load new sample
    inputs =  segment_dataset.data[{{t, math.min(t + batchSize - 1, segment_dataset.size())},{},{},{}}]

    inputs = inputs:cuda()

	prediction[{{t, math.min(t + batchSize - 1, segment_dataset.size())}}] = model:forward(inputs)[{{},{2}}]
end

cutorch.synchronize()

prediction          = prediction:float()
prediction          = torch.round(torch.exp(prediction))
prediction_reshaped = torch.reshape(prediction, opt.height, opt.width)

----------------------------------------------------------------------
-- 						Save the segmentation results 
----------------------------------------------------------------------
print("Saving the segmentation results in " .. opt.predictedDataset)

local f = hdf5.open(opt.predictedPath,'a')
f:write(opt.predictedDataset, prediction_reshaped)
f:close()


