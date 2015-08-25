----------------------------------------------------------------------
print '==> define parameters'

-- hidden units, filter sizes (for ConvNet only):
nfeaturemaps  = { {{ nfeaturemaps|join(', ') }} }
filtsize 	  = {{ filtsize }}
poolsize 	  = { {{ poolsize|join(', ') }} }
featuremaps_h = {{ featuremaps_h }}
featuremaps_w = {{ featuremaps_w }}
noutputs 	  = {{ noutputs }}

----------------------------------------------------------------------
print '==> construct model'
model = nn.Sequential()

-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(cudnn.SpatialConvolution(nfeats, nfeaturemaps[1], filtsize, filtsize))
model:add(cudnn.{{ activation_function }}(true))
model:add(cudnn.{{ pooling_function }}(poolsize[1],poolsize[1],poolsize[1],poolsize[1]))
-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(cudnn.SpatialConvolution(nfeaturemaps[1], nfeaturemaps[2], filtsize, filtsize))
model:add(cudnn.{{ activation_function }}(true))
model:add(cudnn.{{ pooling_function }}(poolsize[2],poolsize[2],poolsize[2],poolsize[2]))
-- stage 2 : standard 2-layer MLP:
model:add(nn.View(nfeaturemaps[2]*featuremaps_h*featuremaps_w))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nfeaturemaps[2]*featuremaps_h*featuremaps_w, nfeaturemaps[3]))
model:add(nn.{{ activation_function }}())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nfeaturemaps[3], nfeaturemaps[4]))
model:add(nn.{{ activation_function }}())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nfeaturemaps[4], noutputs))

model:add(nn.LogSoftMax())

-- 32*32 -> 28*28 -> 14*14

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
criterion = nn.ClassNLLCriterion()
