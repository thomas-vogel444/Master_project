require 'utils.lua'
require 'nn'
require 'torch'
require 'optim'
require 'xlua'
require 'cunn'
require 'fbcunn'
require 'fbnn'
require 'cudnn'

print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('CNN for CT scan')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-GPU_id', 1, "Which GPU to use")
cmd:option('-number_of_GPUs', 1, "Which GPU to use")
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 4, 'number of threads')
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
cmd:text()
opt = cmd:parse(arg or {})

print('==> switching to CUDA')
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.GPU_id)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all'

dofile 'data.lua'
dofile(opt.modelPath)
dofile 'train.lua'
dofile 'test.lua'

----------------------------------------------------------------------
print '==> training!'

maxEpoch = opt.maxepoch
while true do
    train()
    test()
    if epoch > maxEpoch then
    	break
    end
end

----------------------------------------------------------------------
local filename = paths.concat(opt.savingDirectory, 'model.net')

print('==> saving model to '..filename)
os.execute('mkdir -p ' .. sys.dirname(filename))
torch.save(filename, model)
