----------------------------------------------------------------------
print '==> define parameters'

-- Define the number of classes needed
if opt.size == 'reduced' then
	noutputs = 2
else
	noutputs = 10
end

-- input dimensions
nfeats = 1
width = 32
height = 32
ninputs = nfeats*width*height

-- hidden units, filter sizes (for ConvNet only):
nfeaturemaps = {64,64,128}
filtsize = 5
poolsize = 2

----------------------------------------------------------------------
print '==> construct model'

model = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling
model:add(nn.SpatialConvolutionMM(nfeats, nfeaturemaps[1], filtsize, filtsize))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 2 : filter bank -> squashing -> L2 pooling
model:add(nn.SpatialConvolutionMM(nfeaturemaps[1], nfeaturemaps[2], filtsize, filtsize))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

-- stage 3 : standard 2-layer neural network
model:add(nn.Reshape(nfeaturemaps[2]*filtsize*filtsize))
model:add(nn.Linear(nfeaturemaps[2]*filtsize*filtsize, nfeaturemaps[3]))
model:add(nn.ReLU())
model:add(nn.Linear(nfeaturemaps[3], noutputs))

model:add(nn.LogSoftMax())

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
criterion = nn.ClassNLLCriterion()