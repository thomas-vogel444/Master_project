----------------------------------------------------------------------
-- CUDA?
if opt.type == 'cuda' then
	model:cuda()
	criterion:cuda()
end

----------------------------------------------------------------------
print '==> defining some tools'

classes = {'0','1'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
	parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'

optimState = {
	learningRate = opt.learningRate,
	weightDecay = opt.weightDecay,
	momentum = opt.momentum,
	learningRateDecay = 1e-7
}
optimMethod = optim.sgd

----------------------------------------------------------------------
print '==> defining training procedure'

function train()

    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local time = sys.clock()

    -- set model to training mode (for modules that differ in training and testing, like Dropout)
    model:training()

    -- shuffle at each epoch
    if opt.size == "small" then
        trainingSize = 2500
    elseif opt.size == "full" then
        trainingSize = trainData.size()
    end

    shuffle = torch.randperm(trainData.size())

    -- do one epoch
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    for t = 1,trainingSize,opt.batchSize do
    	-- disp progress
    	xlua.progress(t, trainingSize)

    	-- create mini batch
        batchSize = math.min(opt.batchSize,trainingSize - t + 1)

    	inputs = torch.Tensor(batchSize,nfeats,width,height)
    	targets = torch.Tensor(batchSize)
    	for i = t,math.min(t+opt.batchSize-1,trainingSize) do
        	-- load new sample
        	inputs[{{i%batchSize + 1},{},{},{}}] = trainData.data[shuffle[i]]
            targets[i%batchSize + 1]             = trainData.labels[shuffle[i]]
    	end

        if opt.type == 'cuda' then 
            inputs    = inputs:cuda() 
            targets   = targets:cuda()
        end

    	-- create closure to evaluate f(X) and df/dX
    	local feval = function(x)
            -- get new parameters
            if x ~= parameters then
            	parameters:copy(x)
            end

            -- reset gradients
            gradParameters:zero()

            -- f is the average of all criterions
            local f = 0

            -- evaluate function for complete mini batch
            for i = 1,inputs:size()[1] do
            	-- estimate f
            	local output = model:forward(inputs[i])
                local err = criterion:forward(output, targets[i])

            	f = f + err

            	-- estimate df/dW
            	local df_do = criterion:backward(output, targets[i])
            	model:backward(inputs[i], df_do)

            	-- update confusion
            	confusion:add(output, targets[i])
            end

            -- normalize gradients and f(X)
            gradParameters:div(inputs:size()[1])
            f = f/inputs:size()[1]

            -- return f and df/dX
            return f,gradParameters
        end

    	-- optimize on current mini-batch
    	optimMethod(feval, parameters, optimState)
	end

	-- time taken
    time = sys.clock() - time
    time = time / trainingSize
    print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix
    print(confusion)

    -- update logger
    trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}

    -- save/log current net
    local filename = paths.concat(opt.save, 'model.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('==> saving model to '..filename)
    torch.save(filename, model)

    -- next epoch
    confusion:zero()
    epoch = epoch + 1
end