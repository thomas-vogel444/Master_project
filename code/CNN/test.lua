----------------------------------------------------------------------
print '==> defining test procedure'

-- test function
function test()
   -- next iteration:
   confusion:zero()

   -- local vars
   local time = sys.clock()

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   prediction = torch.zeros(testData.size(), 2)
   model:evaluate()
--   model:cuda()
   prediction = prediction:cuda()
   
   -- test over test data
   print('==> testing on test set:') 

   for t = 1,testData.size(),opt.batchSize do
       -- disp progress
       xlua.progress(math.min(t + opt.batchSize -1, testData.size()), testData.size())

       if t < (testData.size() - opt.batchSize) then
                batchSize = opt.batchSize
       else
                batchSize = testData.size() - t - math.fmod(testData.size() - t,opt.number_of_GPUs)
       end
       -- load new sample
       local inputs = testData.data[{{t, math.min(t + batchSize - 1, testData.size())},{},{},{}}]

      inputs = inputs:cuda()

      -- test sample
      prediction[{{t, math.min(t + batchSize - 1, testData.size())}}] = model:forward(inputs)

   end

   if opt.number_of_GPUs > 1 then cutorch.synchronize() end

  targets = testData.labels:float()
  prediction = prediction:float()
  confusion:batchAdd(prediction, targets)

   -- timing
   time = sys.clock() - time
   time = time / targets:size()[1]
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
end
