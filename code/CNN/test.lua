----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. Nothing fancy here...
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
   -- next iteration:
   confusion:zero()

   -- local vars
   local time = sys.clock()

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:') 

   local batchSize = 512 * opt.number_of_GPUs

   for t = 1,testData.size(),batchSize do
      -- disp progress
      xlua.progress(math.min(t + batchSize -1, testData.size()), testData.size())

       -- load new sample
       local inputs = testData.data[{{t, math.min(t + batchSize - 1, testData.size())},{},{},{}}]
       local targets = testData.labels[{{t, math.min(t + batchSize - 1, testData.size())}}]

       if opt.type == 'cuda' then
           inputs = inputs:cuda()
       end

      -- test sample
      prediction[{{t, math.min(t + batchSize - 1, testData.size())}}] = model:forward(inputs)[{{},{2}}]

      confusion:batchAdd(prediction, targets)
   end


   if opt.num_gpu > 1 then cutorch.synchronize() end

   -- timing
   time = sys.clock() - time
   time = time / testData.size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
end
