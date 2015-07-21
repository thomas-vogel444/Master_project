require "nn"
require "torch"
require "xlua"
require "hdf5"
require "image"
require 'cunn'

filename = "../../datasets/segmentation_datasets.hdf5"
print("Loading segmentation data from: " .. filename)

----------------------------------------------------------------------
-- 						Segmentation Dataset
----------------------------------------------------------------------
-- Loading the segmentation dataset
local f = hdf5.open(filename,'r')
data = f:read("segmentation_dataset"):all():float()

segmentDataset = {}
segmentDataset.data = data
segmentDataset.size = function() return(segmentDataset.data:size()[1]) end

-- Normalize the data
segmentDataset.data:add(-segmentDataset.data:mean())
segmentDataset.data:div(segmentDataset.data:std())

-- Classify every voxel in the segmentation dataset
model_path = "model.net"
print("Segmenting the image using the model in " .. model_path)
model = torch.load(model_path)
prediction = torch.zeros(segmentDataset.size())
model:evaluate()

for t = 1,segmentDataset.size() do
  -- disp progress
  xlua.progress(t, segmentDataset.size())

  -- get new sample
  local input = segmentDataset.data[t]

  -- test sample
  prediction[t] = model:forward(input)[2]
end

prediction = torch.round(torch.exp(prediction)) + 1

----------------------------------------------------------------------
-- 						True Labeling
----------------------------------------------------------------------
-- Load the labels
print("Loading the true labels of the segmented image...")
labels = f:read("segmentation_labels"):all():float() + 1

----------------------------------------------------------------------
-- 						Real Values
----------------------------------------------------------------------
-- Load the real values
print("Loading the segmented image...")
values = f:read("segmentation_values"):all():float()
values:div(255)
f:close()

----------------------------------------------------------------------
-- 				Create a RGB image from the values
----------------------------------------------------------------------
local rows, cols = values:size(1), values:size(2)
prediction_2d = torch.reshape(prediction, rows, cols)
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
-- 						Blend the RGB image with the mask
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
image.save("/Users/thomasvogel/Desktop/img.png", img)
