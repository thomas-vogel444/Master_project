require 'utils.lua'
require 'nn'
require 'torch'
require 'optim'
require 'xlua'

print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('CNN for CT scan')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-GPU', 1, "Which GPU to use")
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
-- data:
cmd:option('-dataset', '../../datasets/CNN_datasets.hdf5', 'Dataset path')
-- model:
cmd:option('-modelPath', 'model.lua', 'Model file name')
-- training:
cmd:option('-maxepoch', 30, 'Maximum number of epoch on which to train the NN')
cmd:option('-savingDirectory', 'results', 'subdirectory to save/log experiments in')
cmd:option('-learningRate', 0.01, 'learning rate at t=0')
cmd:option('-batchSize', 512, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0.000, 'weight decay (SGD only)')
cmd:option('-momentum', 0.000, 'momentum (SGD only)')
cmd:option('-type', 'float', 'type: double | float | cuda')
cmd:text()
opt = cmd:parse(arg or {})

-- nb of threads and fixed seed (for repeatable experiments)
print('==> switching to floats')
torch.setdefaulttensortype('torch.FloatTensor')

if opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
   cutorch.setDevice(opt.GPU)
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all'

dofile 'data.lua'
dofile(opt.modelPath)
dofile 'train.lua'
dofile 'test.lua'

----------------------------------------------------------------------
-- print '==> training!'

maxEpoch = opt.maxepoch
while true do
    train()
    test()
    if epoch > maxEpoch then
    	break
    end
end