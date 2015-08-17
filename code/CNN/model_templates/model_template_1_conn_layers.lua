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
model:add(cudnn.SpatialMaxPooling(poolsize[1],poolsize[1],poolsize[1],poolsize[1]))
-- stage 2 : standard 1-layer MLP:
model:add(nn.View(nfeaturemaps[1]*featuremaps_h*featuremaps_w))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nfeaturemaps[1]*featuremaps_h*featuremaps_w, nfeaturemaps[2]))
model:add(nn.{{ activation_function }}())
model:add(nn.Dropout(0.5))
model:add(nn.Linear(nfeaturemaps[2], noutputs))
model:add(nn.LogSoftMax())

-- 32*32 -> 28*28 -> 14*14

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
criterion = nn.ClassNLLCriterion()
