require 'modules/GuidedReLU'
require 'modules/DeconvnetReLU'
require 'modules/GradientRescale'


local dqn = {}

local nql = torch.class('dqn.NeuralQLearnerMod', dqn)


function nql:__init(args)

    print("Creating Neural Q Learner for heatmap + fpv")

    

    self.state_dim  = args.state_dim -- State dimensionality.
    self.actions    = args.actions
    self.n_actions  = #self.actions
    self.verbose    = args.verbose
    self.best       = args.best

    --- epsilon annealing
    self.ep_start   = args.ep or 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = args.ep_end or self.ep
    self.ep_endt    = args.ep_endt or 1000000

    ---- learning rate annealing
    self.lr_start       = args.lr or 0.01 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost.
    self.minibatch_size = args.minibatch_size or 1
    self.valid_size     = args.valid_size or 500

    --- Q-learning parameters
    self.discount       = args.discount or 0.99 --Discount factor.
    self.update_freq    = args.update_freq or 1
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 0
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000
    self.hist_len       = args.hist_len or 1
    self.rescale_r      = args.rescale_r
    self.max_reward     = args.max_reward
    self.min_reward     = args.min_reward
    self.clip_delta     = args.clip_delta
    self.target_q       = args.target_q
    self.bestq          = 0

    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.hist_len*self.ncols, 84, 84}
    self.preproc        = args.preproc  -- name of preprocessing network
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1
    self.bufferSize     = args.bufferSize or 512

    

    self.heatmap_network_type = args.heatmap_network_type

    self.printQLearnFlag = true
    if args.name ~= nil then
    self.name = args.name
    else
    self.name = 'Q Learner'
    end

    self.bestNetworkEpoch = 1

    self.transition_params = args.transition_params or {}

    self.network    = args.network or self:createNetwork()
    
    pcall(require,'cutorch')
    if cutorch.getDevice() ~= args.gpu then
    cutorch.setDevice(args.gpu)
    end

    --- general setup
    torch.setnumthreads(4)

    --- set up random number generators
    -- removing lua RNG; seeding torch RNG with opt.seed and setting cutorch
    -- RNG seed to the first uniform random int32 from the previous RNG;
    -- this is preferred because using the same seed for both generators
    -- may introduce correlations; we assume that both torch RNGs ensure
    -- adequate dispersion for different seeds.

    math.random = nil

    torch.manualSeed(1)
    if args.verbose >= 1 then
        print('Torch Seed:', torch.initialSeed())
    end
    local firstRandInt = torch.random()
    if args.gpu >= 0 then
        cutorch.manualSeed(firstRandInt)
        if args.verbose >= 1 then
            print('CUTorch Seed:', cutorch.initialSeed())
        end
    end

    -- check whether there is a network file
    local network_function
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
              " is not a string!")
    end


    i, j = string.find(self.network, 't7')
    k, l = string.find(self.network, 'apprenticeship')
    s, u = string.find(self.network, 'pretrained')

    if i ~= nil then
    print('Loading network from agent file')
    
    local dqn = dofile('dqn/NeuralQLearner.lua')
    local agent_file = torch.load("../../../../" + self.network)

    self.network = agent_file.network:clone()
    print(self.network)

    elseif k ~= nil or s ~= nil then

    print('Loading saved network')
    pcall(require , 'nn')
    pcall(require, 'cunn')  
    local network_path = '../' .. self.network
    local err, model = pcall(torch.load, self.network)
    self.network = model
    print(self.network)

    else
        local msg, err = pcall(require, self.network)

        if not msg then
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, self.network)
        if not err_msg then
            error("Could not find network file ")
        end
        if self.best and exp.best_model then
            print('Loading best network')
            self.network = exp.best_model
        else
            print('Loading Network')
            self.network = exp.model
        end
        else
        print('Creating Agent Network from ' .. self.network)
        self.network = err
        self.network = self:network()
        end
    end

    
    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
    else
        self.network:float()
    end

    -- Load preprocessing network.
    if self.preproc then
        if not (type(self.preproc == 'string')) then
            error('The preprocessing is not a string')
        end
        msg, err = pcall(require, self.preproc)
        if not msg then
            error("Error loading preprocessing net")
        end
        self.preproc = err
        self.preproc = self:preproc()
        self.preproc:float()
    end

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.tensor_type = torch.FloatTensor
    end

    -- saliency stuff

    self.saliency       = args.saliency or 'normal'
    self:setSaliency(self.saliency)
    self.saliencyMap    = self.tensor_type(1, self.input_dims[2], self.input_dims[3]):zero()
    self.inputGrads     = self.tensor_type(self.histLen*self.input_dims[1], self.input_dims[2], self.input_dims[3]):zero() 

    self.heads = 1 
    self.head = 1

    -- end of saliency stuff

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Foat or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize, heatmap_network_type = self.heatmap_network_type
    }

    self.transitions = args.TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
    self.lastHeatmap = nil
    self.lastAction = nil
    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.

    self.q_max = 1
    self.r_max = 1

    self.w, self.dw = self.network:getParameters()
    self.dw:zero()

    self.deltas = self.dw:clone():fill(0)

    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)

    if self.target_q then
        self.target_network = self.network:clone()
    end

    
end

function nql:getTransitionSize()
    
    return self.transitions:size(0)
end

function nql:updateBestNetwork()
    self.best_network = self.network:clone()
end

function nql:loadBestNetwork()
    self.network = self.best_network:clone()
end

function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    self.network = state.model
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.numSteps = 0
    print("RESET STATE SUCCESFULLY")
end


function nql:preprocess(rawstate)

    if self.preproc then
        return self.preproc:forward(rawstate:float()):clone():reshape(self.state_dim)
    else
    return rawstate:clone():float():reshape(self.state_dim)
    end

end


function nql:getQUpdate(args)

    local s, a, r, s2, term, delta
    local q, q2, q2_max

    s = args.s
    a = args.a
    r = args.r
    s2 = args.s2
    term = args.term

    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    term = term:clone():float():mul(-1):add(1)

    local target_q_net
    if self.target_q then
        target_q_net = self.target_network
    else
        target_q_net = self.network
    end

    -- Compute max_a Q(s_2, a).
    q2_max = target_q_net:forward(s2):float():max(2)

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    q2 = q2_max:clone():mul(self.discount):cmul(term)

    delta = r:clone():float()

    if self.rescale_r then
        delta:div(self.r_max)
    end
    delta:add(q2)

    -- q = Q(s,a)
    local q_all = self.network:forward(s):float()
    q = torch.FloatTensor(q_all:size(1))
    for i=1,q_all:size(1) do
        q[i] = q_all[i][a[i]]
    end
    delta:add(-1, q)

    if self.clip_delta then
        delta[delta:ge(self.clip_delta)] = self.clip_delta
        delta[delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local targets = torch.zeros(self.minibatch_size, self.n_actions):float()
    for i=1,math.min(self.minibatch_size,a:size(1)) do
        targets[i][a[i]] = delta[i]
    end

    if self.gpu >= 0 then targets = targets:cuda() end

    return targets, delta, q2_max
end


function nql:qLearnMinibatch()
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    assert(self.transitions:size() > self.minibatch_size)

    local s, a, r, s2, term = self.transitions:sample(self.minibatch_size)

    local targets, delta, q2_max = self:getQUpdate{s=s, a=a, r=r, s2=s2,
        term=term, update_qmax=true}

    -- zero gradients of parameters
    self.dw:zero()

    -- get new gradient
    self.network:backward(s, targets)

    -- add weight cost to gradient
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    -- use gradients
    self.g:mul(0.95):add(0.05, self.dw)
    self.tmp:cmul(self.dw, self.dw)
    self.g2:mul(0.95):add(0.05, self.tmp)
    self.tmp:cmul(self.g, self.g)
    self.tmp:mul(-1)
    self.tmp:add(self.g2)
    self.tmp:add(0.01)
    self.tmp:sqrt()

    -- accumulate update
    self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
    self.w:add(self.deltas)
end


function nql:sample_validation_data()
    print('Sampling validation data for '.. self.name)
    local s, a, r, s2, term = self.transitions:sample(self.valid_size)
    self.valid_s    = s:clone()
    self.valid_a    = a:clone()
    self.valid_r    = r:clone()
    self.valid_s2   = s2:clone()
    self.valid_term = term:clone()
end


function nql:compute_validation_statistics()

    if not self.valid_s then
    self:sample_validation_data()
    end

    local targets, delta, q2_max = self:getQUpdate{s=self.valid_s, a=self.valid_a, r=self.valid_r, s2=self.valid_s2, term=self.valid_term}

    self.v_avg = self.q_max * q2_max:mean()
    self.tderr_avg = delta:clone():abs():mean()
end

function nql:getEpsilon()
    return self.ep
end

----- saliency code

function nql:setSaliency(saliency)
    self.saliency = saliency
    self:modelSetSaliency(saliency)
end

function nql:modelSetSaliency(saliency)

    self.saliency = saliency
    local relus, relucontainers = self.target_network:findModules(hasCudnn and 'cudnn.ReLU' or 'nn.ReLU')
      if #relus == 0 then
        relus, relucontainers = self.target_network:findModules('nn.GuidedReLU')
      end
      if #relus == 0 then
        relus, relucontainers = self.target_network:findModules('nn.DeconvnetReLU')
      end

      -- Work out which ReLU to use now
      local layerConstructor = hasCudnn and cudnn.ReLU or nn.ReLU
      self.relus = {} --- Clear special ReLU list to iterate over for salient backpropagation
      if saliency == 'guided' then
        layerConstructor = nn.GuidedReLU
      elseif saliency == 'deconvnet' then
        layerConstructor = nn.DeconvnetReLU
      end

      -- Replace ReLUs
      for i = 1, #relus do
        -- Create new special ReLU
        local layer = layerConstructor()

        -- Copy everything over
        for key, val in pairs(relus[i]) do
          layer[key] = val
        end

        -- Find ReLU in containing module and replace
        for j = 1, #(relucontainers[i].modules) do
          if relucontainers[i].modules[j] == relus[i] then
            relucontainers[i].modules[j] = layer
          end
        end
      end

      -- Create special ReLU list to iterate over for salient backpropagation
      self.relus = self.target_network:findModules(saliency == 'guided' and 'nn.GuidedReLU' or 'nn.DeconvnetReLU')

end

function nql:modelSalientBackprop()
  for i, v in ipairs(self.relus) do
    v:salientBackprop()
  end
end

-- Switches the backward computation of special ReLUs for normal backpropagation
function nql:modelNormalBackprop()
  for i, v in ipairs(self.relus) do
    v:normalBackprop()
  end
end

function nql:getSaliencyMap(display)
  local screen = display:clone() -- Cloned to prevent side-effects
  local saliencyMap = agent.saliencyMap:float()

  -- Use red channel for saliency map
  screen:select(1, 1):copy(image.scale(saliencyMap, self.input_dims[2], self.input_dims[3]))

  return screen
end

function nql:computeSaliency(state, index, ensemble)
  -- Switch to possibly special backpropagation
  self:modelSalientBackprop()

  -- Create artificial high target
  local maxTarget = self.Tensor(self.heads, self.n_actions):zero()
  if ensemble then
    -- Set target on all heads (when using ensemble policy)
    maxTarget[{{}, {index}}] = 1
  else
    -- Set target on current head
    maxTarget[self.head][index] = 1
  end

  -- Backpropagate to inputs
  self.inputGrads = self.network:backward(state, maxTarget)
  -- Saliency map ref used by Display
  self.saliencyMap = torch.abs(self.inputGrads:select(1, self.recurrent and 1 or self.histLen):float())

  -- Switch back to normal backpropagation
  self:modelNormalBackprop()
end

------ end of saliency stuff

function nql:perceive(reward, rawstate, heatmap,  terminal, testing, testing_ep)

    -- Preprocess state (will be set to nil if terminal)
    local state = self:preprocess(rawstate):float()
    local heatmap = self:preprocess(heatmap):float()

    local curState

    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, reward)
    end

    self.transitions:add_recent_state(state, heatmap, terminal)

    local currentFullState = self.transitions:get_recent()

    --Store transition s, a, r, s'
    if self.lastState and not testing then
        self.transitions:add(self.lastState, self.lastHeatmap, self.lastAction, reward,self.lastTerminal, priority)
    end


    if self.numSteps == self.learn_start+1 and not testing then
        self:sample_validation_data()
    end

    curState= self.transitions:get_recent()
    curState = curState:resize(1, unpack(self.input_dims))

    -- Select action
    local actionIndex = 1
    if not terminal then
        actionIndex = self:eGreedy(curState, testing_ep)
    end

    --saliency generation
    if testing then
        self:computeSaliency(curState, actionIndex, false)
    else
        self.saliencyMap:zero()
    end

    self.transitions:add_recent_action(actionIndex)

    --Do some Q-learning updates
    if self.numSteps > self.learn_start and not testing and self.numSteps % self.update_freq == 0  then

        if self.printQLearnFlag == true then
            print('Started Q updates for ' .. self.name)
            self.printQLearnFlag = false
        end

        for i = 1, self.n_replay do
            self:qLearnMinibatch()
        
        end
    end

    if not testing then
        self.numSteps = self.numSteps + 1
    end

    self.lastHeatmap = heatmap:clone()
    self.lastState = state:clone()
    self.lastAction = actionIndex
    self.lastTerminal = terminal

    if self.target_q and self.numSteps % self.target_q == 1 then
        self.target_network = self.network:clone()
    end

    if not terminal then
        return actionIndex
    else
        return 0
    end
end


function nql:eGreedy(state, testing_ep)

    local currentEp = nil
    if testing_ep then 
        currentEp = testing_ep
    else
        self.ep =  (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    currentEp = self.ep
    end 

    assert(currentEp, 'Ep cannot be nil')
    
    -- Epsilon greedy
    if torch.uniform() < currentEp then
        return torch.random(1, self.n_actions)
    else
        return self:greedy(state)
    end
end


function nql:greedy(state)
    if self.gpu >= 0 then
        state = state:cuda()
    end

    local q = self.network:forward(state):float():squeeze()
    local maxq = q[1]
    local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, self.n_actions do
        if q[a] > maxq then
            besta = { a }
            maxq = q[a]
        elseif q[a] == maxq then
            besta[#besta+1] = a
        end
    end
    self.bestq = maxq

    local r = torch.random(1, #besta)

    self.lastAction = besta[r]

    return besta[r]
end


function nql:createNetwork()
    local n_hid = 128
    local mlp = nn.Sequential()
    mlp:add(nn.Reshape(self.hist_len*self.ncols*self.state_dim))
    mlp:add(nn.Linear(self.hist_len*self.ncols*self.state_dim, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))

    return mlp
end


function nql:_loadNet()
    local net = self.network
    if self.gpu then
        net:cuda()
    else
        net:float()
    end
    return net
end


function nql:init(arg)
    self.actions = arg.actions
    self.n_actions = #self.actions
    self.network = self:_loadNet()
    -- Generate targets.
    self.transitions:empty()
end

local function recursive_map(module, field, func)
    local str = ""
    if module[field] or module.modules then
        str = str .. torch.typename(module) .. ": "
    end
    if module[field] then
        str = str .. func(module[field])
    end
    if module.modules then
        str = str .. "["
        for i, submodule in ipairs(module.modules) do
            local submodule_str = recursive_map(submodule, field, func)
            str = str .. submodule_str
            if i < #module.modules and string.len(submodule_str) > 0 then
                str = str .. " "
            end
        end
        str = str .. "]"
    end

    return str
end

local function abs_mean(w)
    return torch.mean(torch.abs(w:clone():float()))
end

local function abs_max(w)
    return torch.abs(w:clone():float()):max()
end

-- Build a string of average absolute weight values for the modules in the
-- given network.
local function get_weight_norms(module)
    return "Weight norms:\n" .. recursive_map(module, "weight", abs_mean) ..
            "\nWeight max:\n" .. recursive_map(module, "weight", abs_max)
end

-- Build a string of average absolute weight gradient values for the modules
-- in the given network.
local function get_grad_norms(module)
    return "Weight grad norms:\n" ..
        recursive_map(module, "gradWeight", abs_mean) ..
        "\nWeight grad max:\n" .. recursive_map(module, "gradWeight", abs_max)
end

function nql:report()
    print(get_weight_norms(self.network))
    print(get_grad_norms(self.network))
end

return dqn