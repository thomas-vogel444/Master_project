require 'nn'
require 'torch'
require 'optim'
require 'xlua'

print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST Loss Function')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
cmd:option('-identifier', 1, 'integer identifying the NN for later procesing')
-- data:
cmd:option('-size', 'full', 'how many samples do we load: reduced | small | full')
-- model:
cmd:option('-model', 'convnet', 'type of model to construct: mlp | convnet')
-- training:
cmd:option('-maxepoch', 15, 'Maximum number of epoch on which to train the NN')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-learningRate', 0.1, 'learning rate at t=0')
cmd:option('-batchSize', 128, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0.001, 'weight decay (SGD only)')
cmd:option('-momentum', 0.001, 'momentum (SGD only)')
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

----------------------------------------------------------------------
print '==> executing all'

dofile 'data.lua'

if opt.model == 'convnet' then 
	dofile 'model_convnet.lua'
elseif opt.model == 'convnet2' then
	dofile 'model_convnet2.lua'
elseif opt.model == 'mlp' then
	dofile 'model_mlp.lua'
end

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

-- Appending the relevant result into a file 
f = io.open('results/results.txt', 'a')
f:write('% mean class accuracy for NN #' .. opt.identifier .. ": " .. confusion.totalValid .. "\n")
f:close()









