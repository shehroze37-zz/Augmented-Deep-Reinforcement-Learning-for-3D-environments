require 'torch'
require 'math'
require 'nn'
require 'optim'
require 'gnuplot'
require 'sys'
npy4th = require 'npy4th'
require 'xlua'
require 'cunn'


torch.manualSeed(12)
torch.setnumthreads(4)
torch.setdefaulttensortype('torch.FloatTensor')


opt = {
  nonlinearity_type = 'sigmoid',
  batch_size = 32, 
  epochs = 1000,  
  print_every = 25, 
  gpu = 1,
  load_saved_model = 'apprenticeship_model.net',
  save = 'apprenticeship-models/',
  coefL1 = 0,
  coefL2 = 1
  
}

optimState = {
	learningRate = 1e-2,
	weightDecay = 0,
	momentum = 0,
	learningRateDecay = 0}

possible_actions = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}
classes = {'1','2','3','4','5','6','7','8','9','10','11', '12','13','14','15','16', '17', '18', '19', '20'}

function create_model(args)
  
 
  if args.load_saved_model then
	  
	  print('Loading checkpoint model')
          network  = torch.load(args.load_saved_model)
	  print(network)
	  return network

  else
  	  local convolutional_network_file = dofile('convnet_2.lua')
	  args = {}
	  args['preproc_atari'] = "preproc_atari"
	  args['gpu'] = 1	  
	  input_dimensions = {4,120,120} 
	  args['input_dims'] = input_dimensions 
	  args['state_dim'] = 120 * 120  

	  args['n_actions'] =  #possible_actions
	  args['verbose'] = 2


	  local network = convolutional_network_file(args)
	  
	  return network

  end 

  
end


function load_data()

  --data divided 80/20
  local folder_name = 'apprenticeship-data/' 

  local states = {}
  local actions = {}

  for i = 1, 118 do

    local network_input = npy4th.loadnpy(folder_name .. i - 1 .. '/network_input_heatmap_120')
    local network_actions = npy4th.loadnpy(folder_name .. i - 1 .. '/actions')

    if network_input:size(1) ~= network_actions:size(1) or network_actions:size(2) ~= 1 then
      print('ERROR')
      return nil
    else
      for j = 1, network_input:size(1) do
	if network_input[j]:sum() ~= 0 and network_actions[j] ~= 0 then
		states[#states + 1] = network_input[j]
        	actions[#actions + 1] = network_actions[j]
	end
      end
    end

  end

  local upperBound = math.ceil(#states * 0.8)
  local training_data_states = torch.Tensor(upperBound, 4, 120 * 120)
  local training_data_actions = torch.Tensor(upperBound)

  local testing_data_states = torch.Tensor(#states - upperBound , 4, 120 * 120)
  local testing_data_actions = torch.Tensor(#states - upperBound)

  print('Training data size = ' ..  training_data_states:size(1))
  print('Testing  data size = ' .. testing_data_actions:size(1))

 
  local indices = torch.randperm(#states)

  local testing_size = 0
  for i = 1, #states do 

    if i <= upperBound then
      training_data_states[i] = states[indices[i]]
      training_data_actions[i] = actions[indices[i]]
    else
      testing_size  = testing_size + 1
      testing_data_states[testing_size] = states[indices[i]]
      testing_data_actions[testing_size] = actions[indices[i]]
    end

  end

  print('Loaded Dataset with testing size ' .. testing_size)
  return training_data_states, training_data_actions, testing_data_states, testing_data_actions
  
end


training_accuracy = {}
test_accuracy = {}
train_dataset_states, train_dataset_actions, test_dataset_states, test_dataset_actions = load_data()


print(train_dataset_actions:eq(0):sum())
print(test_dataset_actions:eq(0):sum())


train_stat = torch.Tensor(#possible_actions)
print("train")
for i=1,20 do
	local nb_occurence = train_dataset_actions:eq(i):sum()
	nb_occurence = math.max(10, nb_occurence)
	train_stat[i] = 10/nb_occurence
	print(i..": "..nb_occurence)
end


model = create_model(opt)
--model:add(nn.LogSoftMax())
--criterion = nn.ClassNLLCriterion()
criterion = nn.MSECriterion()
model:cuda()
criterion:cuda()
params, grads = model:getParameters()

training_confusion = optim.ConfusionMatrix(classes)
testing_confusion = optim.ConfusionMatrix(classes)


trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
trainingBatchLossLogger = optim.Logger(paths.concat(opt.save, 'training_loss.log'))

function train(model, criterion, args, data_states, data_actions)

    print('Started training')
    local counter = 0
    local time = sys.clock()

    --[[ (re-)initialize weights
    params:uniform(-0.01, 0.01)
    if opt.nonlinearity_type == 'requ' then
        -- need to offset bias for requ/relu/etc s.t. we're at x > 0 (so dz/dx is nonzero)
        for _, lin in pairs(model:findModules('nn.Linear')) do
            lin.bias:add(0.5)
        end
    end]]

    local indices = torch.randperm(data_states:size(1))
    local current_batch = 1
    local batch_loss = 0 
    local gt = torch.zeros(args.batch_size, #possible_actions):cuda()
    local inp  = torch.CudaTensor(32,4,120*120)


    local feval = function(x)
      
      if x ~= params then
        params:copy(x)
      end

      local batch_indices = indices:narrow(1, current_batch, args.batch_size):long()
      current_batch = current_batch + args.batch_size

      local batch_inputs = data_states:index(1, batch_indices)
      inp:copy(batch_inputs)

      local batch_target = data_actions:index(1, batch_indices)

      gt:zero()
      for i=1,args.batch_size do
		gt[i][batch_target[i]] = 1
      end

      grads:zero()
      -- forward
      local outputs = model:forward(inp)
	
      local loss = criterion:forward(outputs, gt)
      batch_loss = batch_loss + loss
      -- backward


      --l2 regularization
      --[[local norm,sign= torch.norm,torch.sign
      loss = loss + args.coefL2 * norm(params,2)^2/2
      grads:add(params:clone():mul(opt.coefL2) )]]

      local actions = getActions(outputs) 
      for i = 1, opt.batch_size do
	       --training_confusion:add(outputs[i], batch_target[i] )
		training_confusion:add(actions[i], batch_target[i])
      end

      local dloss_doutput = criterion:backward(outputs, gt)
      model:backward(inp, dloss_doutput)

      return loss, grads
    end

    local losses = {}

    for i = 1, data_states:size(1) - args.batch_size + 1, args.batch_size do
      xlua.progress(i, data_states:size(1))
      local _, loss = optim.adagrad(feval, params, optimState)
      losses[#losses + 1] = loss[1]
    end

   time = sys.clock() - time
   time = time / data_states:size(1)
   print(" ")
   print("loss: "..batch_loss)
   trainingBatchLossLogger:add{['Training batch loss'] = batch_loss}
   print("Training time for one epoch = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(training_confusion)
   local training_accuracy = training_confusion.totalValid * 100
   trainLogger:add{['% mean class accuracy (train set)'] = training_accuracy}

   training_confusion:zero()


   local filename = paths.concat(opt.save, 'apprenticeship_model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   torch.save(filename, model)	

   return model, losses, training_accuracy
end

local function test(model, criterion, args, data_states, data_actions)

   local time = sys.clock()
   local indices = torch.randperm(data_states:size(1))
   local current_batch = 1
   local inp  = torch.CudaTensor(32,4,120*120)

   print('Testing')
   for t = 1, data_states:size(1) - args.batch_size + 1, args.batch_size do

      xlua.progress(t, data_states:size(1))




      local batch_indices = indices:narrow(1, current_batch, args.batch_size):long()
      current_batch = current_batch + args.batch_size
      local batch_inputs = data_states:index(1, batch_indices)
      local batch_target = data_actions:index(1, batch_indices)

      inp:copy(batch_inputs)	

      -- test samples
      local preds = model:forward(inp)

      -- confusion:
      local actions = getActions(preds) 
      for i = 1, args.batch_size do
         --testing_confusion:add(preds[i], batch_target[i])
	   testing_confusion:add(actions[i], batch_target[i])
      end
   end

   time = sys.clock() - time
   time = time / data_states:size(1)
   print("Testing time = " .. (time*1000) .. 'ms')

   print(testing_confusion)
   testLogger:add{['% mean class accuracy (test set)'] = testing_confusion.totalValid * 100}

   local accuracy = testing_confusion.totalValid * 100
   testing_confusion:zero()

   return accuracy

end

function getActions(output)


    local final_output = torch.CudaTensor(32)

    for i = 1, output:size(1) do 

    	    local q = output[i]:float():squeeze()
	    local maxq = q[1]
	    local besta = {1}

	    for a = 2, #possible_actions do
		if q[a] > maxq then
		    besta = { a }
		    maxq = q[a]
		elseif q[a] == maxq then
		    besta[#besta+1] = a
		end
	    end

	    local r = torch.random(1, #besta)
	    final_output[i] =  besta[r]
    end

    return final_output
end


--sample_test = torch.CudaTensor(32,4,120*120):fill(1)
--local output = model:forward(sample_test)
--print(output[1]:float():squeeze())
--print('Action = ')
--print(getAction(output[1]))

--[[test(model, criterion, opt, test_dataset_states, test_dataset_actions)]]

for i = 100, opt.epochs do

    local _, training_losses, accuracy = train(model, criterion, opt, train_dataset_states, train_dataset_actions)

    training_accuracy[i] = accuracy

    if i % 10 then 
	     local testing_accuracy = test(model, criterion, opt, test_dataset_states, test_dataset_actions)
             test_accuracy[i] = testing_accuracy
    end

    torch.save('training_accuracy', training_accuracy)
    torch.save('testing_accuracy', test_accuracy)

    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    testLogger:style{['% mean class accuracy (test set)'] = '-'}
    trainingBatchLossLogger:style{['Training loss '] = '-'}
    trainLogger:plot()
    testLogger:plot()
    trainingBatchLossLogger:plot()
   
    print('******************************')

end
