----------------------------------------------------------------------
model:cuda()
criterion:cuda()

----------------------------------------------------------------------
print '==> defining some tools'

classes = {'0','1'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.savingDirectory, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.savingDirectory, 'test.log'))

----------------------------------------------------------------------
print '==> configuring optimizer'

optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = 1e-7
}

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

-- Optimizer
optimator = nn.Optim(model, optim_state)

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
if opt.number_of_GPUs > 1 then
    parameters, gradParameters = model:get(1):getParameters()
    cutorch.synchronize()
    model:cuda()  -- get it back on the right GPUs
else
    parameters, gradParameters = model:getParameters()
end

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
    trainingSize = trainData.size()

    shuffle = torch.randperm(trainData.size())

    -- do one epoch
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    for t = 1,trainingSize,opt.batchSize do
        -- disp progress
        xlua.progress(t, math.min(t+opt.batchSize-1,trainingSize))

        -- create mini batch
        batchSize = math.min(opt.batchSize,trainingSize - t + 1)

        inputs = torch.Tensor(batchSize,nfeats,patchsize,patchsize)
        targets = torch.Tensor(batchSize)
        for i = t,math.min(t+opt.batchSize-1,trainingSize) do
            -- load new sample
            inputs[{{i%batchSize + 1},{},{},{}}] = trainData.data[shuffle[i]]
            targets[i%batchSize + 1]             = trainData.labels[shuffle[i]]
        end

        inputs    = inputs:cuda() 
        targets   = targets:cuda()
  
        f, outputs = optimator:optimize(optim.sgd, inputs, targets, criterion)

        if opt.num_gpu > 1 then cutorch.synchronize() end
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
    local filename = paths.concat(opt.savingDirectory, 'model.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('==> saving model to '..filename)

    torch.save(filename, model)

    -- next epoch
    confusion:zero()
    epoch = epoch + 1
end