cmd = torch.CmdLine()
cmd:text()
cmd:text('Segmentation')
cmd:text()
cmd:text('Options:')
cmd:option('-segmentationFile', "../../datasets/segmentation_datasets.hdf5", "Path to the segmentation file")
cmd:option('-segmentationLabels', "segmentation_labels_fixed_z", "Segmentation labels")
cmd:option('-segmentationValues', "segmentation_values_fixed_z", "Segmentation values")
cmd:option('-predictedPath', "predicted_labels.hdf5", 'Path to the predicted file')
cmd:option('-predictedDataset', "predicted_labels_fixed_z", 'Dataset name contaiing the predicted labels')
cmd:option('-imagePath', 'mask_fixed_z')
cmd:option('-modelPath', "model.net", 'Paths to the model file')	
cmd:option('-type', 'float', 'type: float | cuda')
cmd:text()

----------------------------------------------------------------------
-- 						Load predicted values
----------------------------------------------------------------------
-- Loading the segmentation dataset
print("Loading the predicted values " .. opt.segmentationDataset .. " from " .. opt.segmentationFile)
local f = hdf5.open(opt.predictedPath,'r')
prediction_2d = f:read(opt.predictedDataset):all():float()
f:close()

----------------------------------------------------------------------
--            True Labeling
----------------------------------------------------------------------
-- Load the labels and values
print("Loading the true labels and the values of the segmented image...")
local f = hdf5.open(opt.segmentationFile,'r')
labels = f:read(opt.segmentationLabels):all():float()
values = f:read(opt.segmentationValues):all():float()
values:div(255)
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
        if (labels[{r,c}] == 0 and prediction_2d[{r,c}] == labels[{r,c}]) then
            mask[{1,r,c}] = 1
            mask[{2,r,c}] = 0
            mask[{3,r,c}] = 0
        elseif (labels[{r,c}] == 1 and prediction_2d[{r,c}] == labels[{r,c}]) then
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