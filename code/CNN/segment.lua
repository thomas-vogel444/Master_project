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
cmd:option('-GPU', 1, "Which GPU to use")
cmd:option('-segmentationFile', "../../datasets/segmentation_datasets.hdf5", "Path to the segmentation file")
cmd:option('-segmentationDataset', "segmentation_dataset_fixed_z", "Segmentation dataset")
cmd:option('-segmentationLabels', "segmentation_labels_fixed_z", "Segmentation labels")
cmd:option('-segmentationValues', "segmentation_values_fixed_z", "Segmentation values")
cmd:option('-predictedPath', "predicted_labels.hdf5", 'Path to the predicted file')
cmd:option('-predictedDataset', "predicted_labels_fixed_z", 'Dataset name contaiing the predicted labels')
cmd:option('-imagePath', 'mask_fixed_z')
cmd:option('-modelPath', "model.net", 'Paths to the model file')
cmd:option('-type', 'float', 'type: float | cuda')
cmd:text()
opt = cmd:parse(arg or {})

if opt.type == 'cuda' then
    require "cunn"
    cutorch.setDevice(opt.GPU)
end

----------------------------------------------------------------------
-- 						Load dataset
----------------------------------------------------------------------
-- Loading the segmentation dataset
print("Loading the dataset " .. opt.segmentationDataset .. " from " .. opt.segmentationFile)
local f = hdf5.open(opt.segmentationFile,'r')
data = f:read(opt.segmentationDataset):all():float()

----------------------------------------------------------------------
--            True Labeling
----------------------------------------------------------------------
-- Load the labels
print("Loading the true labels of the segmented image...")
labels = f:read(opt.segmentationLabels):all():float()

----------------------------------------------------------------------
--            Real Values
----------------------------------------------------------------------
-- Load the real values
print("Loading the segmented image...")
values = f:read(opt.segmentationValues):all():float()
values:div(255)
f:close()

----------------------------------------------------------------------
--            Segment dataset
----------------------------------------------------------------------
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

-- Transfer to the GPU if appropriate
if opt.type == 'cuda' then
	model:cuda()
	prediction = prediction:cuda()
end

for t = 1,segment_dataset.size() do
  	-- disp progress
  	xlua.progress(t, segment_dataset.size())

  	-- get new sample
    local input = segment_dataset.data[t]

    if opt.type == 'cuda' then
        input = input:cuda()
    end

  	-- test sample
  	prediction[t] = model:forward(input)[2]
end

if opt.type == 'cuda' then
	prediction = prediction:float()
end

prediction = torch.round(torch.exp(prediction)) + 1
rows, cols = values:size(1), values:size(2)
prediction_2d = torch.reshape(prediction, rows, cols)

----------------------------------------------------------------------
-- 						Save the segmentation results 
----------------------------------------------------------------------
print("Saving the segmentation results in " .. opt.predictedDataset)

local f = hdf5.open(opt.predictedPath,'a')
f:write(opt.predictedDataset, prediction_2d)
f:close()

----------------------------------------------------------------------
--        Create a RGB image from the values
----------------------------------------------------------------------
img  = torch.Tensor(3, rows, cols)
  
img[{1, {}, {}}] = values
img[{2, {}, {}}] = values
img[{3, {}, {}}] = values

--------------------------------------------------------------------------------------------------------
-- Create a mask in RBG using each color as an indicator as to whether a given voxel is correctly 
-- or incorrectly classified
--------------------------------------------------------------------------------------------------------
mask = torch.Tensor(3, rows, cols)
for r=1,rows do
  for c=1,cols do
      if (labels[{r,c}] == 1 and prediction_2d[{r,c}] == labels[{r,c}]) then
          mask[{1,r,c}] = 1
          mask[{2,r,c}] = 0
          mask[{3,r,c}] = 0
      elseif (labels[{r,c}] == 2 and prediction_2d[{r,c}] == labels[{r,c}]) then
          mask[{1,r,c}] = 0
          mask[{2,r,c}] = 1
          mask[{3,r,c}] = 0
      else -- ERROR
      mask[{1,r,c}] = 0
      mask[{2,r,c}] = 0
      mask[{3,r,c}] = 1
    end
  end -- END: for c=1,cols do
end

------------------------------------------------------------------
--            Blend the RGB image with the mask
------------------------------------------------------------------
-- out = image1 * (1.0 - alpha) + image2 * alpha
function blendImages(image1, image2, alpha)
  assert(alpha <= 1)
  local img1 = torch.mul(image1, 1.0 - alpha)
  local img2 = torch.mul(image2, alpha)
  local result = torch.add(img1, img2)

  return result
end

img = img:float()
mask = mask:float()
img = blendImages(img, mask, 0.4)
print("Saving the segmented image...")
image.save(opt.imagePath, img)










