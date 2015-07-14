require "nn"

----------------------------------------------------------------------
print '==> define parameters'

-- input dimensions
nfeats  = 6
width   = 32
height  = 32
ninputs = nfeats*width*height

-- hidden units, filter sizes (for ConvNet only):
nfeaturemaps  = {32,64,1000,1000}
filtsize 	  = 5
poolsize 	  = {3,2}
featuremaps_h = 2
featuremaps_w = 2
noutputs 	  = 2

----------------------------------------------------------------------
print '==> construct model'
model = nn.Sequential()

-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMM(nfeats, nfeaturemaps[1], filtsize, filtsize))
model:add(nn.ReLU())
-- model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(poolsize[1],poolsize[1],poolsize[1],poolsize[1]))
-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMM(nfeaturemaps[1], nfeaturemaps[2], filtsize, filtsize))
model:add(nn.ReLU())
-- model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(poolsize[2],poolsize[2],poolsize[2],poolsize[2]))
-- stage 3 : standard 2-layer MLP:
model:add(nn.View(nfeaturemaps[2]*featuremaps_h*featuremaps_w))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nfeaturemaps[2]*featuremaps_h*featuremaps_w, nfeaturemaps[3]))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nfeaturemaps[3], nfeaturemaps[4]))
model:add(nn.ReLU())
-- model:add(nn.Tanh())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nfeaturemaps[3], noutputs))

model:add(nn.LogSoftMax())

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
-- Initialization: for ReLU acivation functions, the parameters should 
-- be initialised as small positive weights
-- size_layer1 = model:get(1):getParameters():size()
-- size_layer2 = model:get(4):getParameters():size()
-- size_layer3 = model:get(8):getParameters():size()
-- size_layer4 = model:get(10):getParameters():size()

----------------------------------------------------------------------
criterion = nn.ClassNLLCriterion()