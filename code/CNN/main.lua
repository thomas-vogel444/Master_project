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
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
cmd:option('-identifier', 1, 'integer identifying the NN for later procesing')
cmd:option('-segment', 'false', 'Boolean indicating whether to segment the test DICOM image after training.')
-- data:
cmd:option('-dataset', '../../datasets/CNN_datasets.hdf5', 'Dataset path')
cmd:option('-size', 'small', 'how many samples do we load: small | full')
-- model:
cmd:option('-model', 'convnet', 'type of model to construct: convnet')
-- training:
cmd:option('-maxepoch', 30, 'Maximum number of epoch on which to train the NN')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-learningRate', 0.01, 'learning rate at t=0')
cmd:option('-batchSize', 128, 'mini-batch size (1 = pure stochastic)')
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
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

-- Records the history of classification errors
classificationErrors = {trainingErrors = {}, testErrors = {}}

----------------------------------------------------------------------
print '==> executing all'

dofile 'data.lua'
dofile 'model.lua'
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

-- Appending the relevant results into a file 
print('==> saving the classification errors')
filename = paths.concat(opt.save, 'results.txt')
f = io.open(filename, 'a')
writeToFile(f, classificationErrors.trainingErrors, "Training")
writeToFile(f, classificationErrors.testErrors, "Testing")
f:close()