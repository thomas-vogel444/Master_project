----------------------------------------------------------------------
print '==> define parameters'

-- hidden units, filter sizes (for ConvNet only):
nfeaturemaps  = { 32, 32, 64, 1000, 500 }
filtsize 	  = 5
poolsize 	  = { 2, 2 }
featuremaps_h = 4
featuremaps_w = 4
noutputs 	  = 2

----------------------------------------------------------------------
print '==> construct model'
model = nn.Sequential()

-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(cudnn.SpatialConvolution(nfeats, nfeaturemaps[1], filtsize, filtsize))
model:add(cudnn.ReLU(true))
-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(cudnn.SpatialConvolution(nfeaturemaps[1], nfeaturemaps[2], filtsize, filtsize))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(poolsize[1],poolsize[1],poolsize[1],poolsize[1]))
-- stage 3 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(cudnn.SpatialConvolution(nfeaturemaps[2], nfeaturemaps[3], filtsize, filtsize))
model:add(cudnn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(poolsize[2],poolsize[2],poolsize[2],poolsize[2]))
-- stage 4 : standard 2-layer MLP:
model:add(nn.View(nfeaturemaps[3]*featuremaps_h*featuremaps_w))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nfeaturemaps[3]*featuremaps_h*featuremaps_w, nfeaturemaps[4]))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nfeaturemaps[4], nfeaturemaps[5]))
model:add(nn.ReLU())
-- model:add(nn.Tanh())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nfeaturemaps[5], noutputs))

model:add(nn.LogSoftMax())

-- 32*32 -> 28*28 -> 24*24 -> 12*12 -> 8*8 -> 4*4

----------------------------------------------------------------------
print '==> here is the model:'
print(model)