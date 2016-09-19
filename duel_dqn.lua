require 'nn'
require 'cunn'
require 'cutorch'
local DuelAggregator = require 'DuelAggregator'


return function(args)

    print('Creating DUEL DQN neural network')

    args.n_units        = {32, 64, 64}
    args.filter_size    = {8, 4, 3}
    args.filter_stride  = {4, 2, 1}
    args.n_hid          = {512}
    args.nl             = nn.ReLU

    print(unpack(args.input_dims))


    local numberOfFrames = 4
    if args.heatmap_network_type == 3 then
        numberOfFrames = 5
    elseif args.heatmap_network_type == 1 or args.heatmap_network_type = 2 then
        numberOfFrames = 8
    end

    args.input_dims[1] = numberOfFrames
    
    local net = nn.Sequential()

    local nested_net = nn.Sequential()
    nested_net:add(nn.View(numberOfFrames, args.input_dims[2], args.input_dims[3])) -- Concatenate history in channel dimension
    nested_net:add(nn.SpatialConvolution(numberOfFrames, 32, 8, 8, 4, 4, 1, 1))
    nested_net:add(nn.ReLU(true))
    nested_net:add(nn.SpatialConvolution(32, 64, 4, 4, 2, 2))
    nested_net:add(nn.ReLU(true))
    nested_net:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1))
    nested_net:add(nn.ReLU(true))

    net:add(nested_net)

    local nel = torch.prod(torch.Tensor(getOutputSize(net, _.append({numberOfFrames}, args.input_dims))))
    net:add(nn.View(nel))

    local head = nn.Sequential()

    -- Value approximator V^(s)
    local valueStream = nn.Sequential()
    valueStream:add(nn.Linear(nel, args.n_hid[1]))
    valueStream:add(nn.ReLU(true))
    valueStream:add(nn.Linear(args.n_hid[1], 1)) -- Predicts value for state

    -- Advantage approximator A^(s, a)
    local advantageStream = nn.Sequential()
    advantageStream:add(nn.Linear(nel, args.n_hid[1]))
    advantageStream:add(nn.ReLU(true))
    advantageStream:add(nn.Linear(args.n_hid[1], args.n_actions)) -- Predicts action-conditional advantage

    -- Streams container
    local streams = nn.ConcatTable()
    streams:add(valueStream)
    streams:add(advantageStream)

    -- Network finishing with fully connected layers
    head:add(nn.GradientRescale(1/math.sqrt(2), true)) -- Heuristic that mildly increases stability for duel
    -- Create dueling streams
    head:add(streams)
    -- Add dueling streams aggregator module
    head:add(DuelAggregator(args.n_actions))


    local headConcat = nn.ConcatTable()
    headConcat:add(head)
    net:add(headConcat)

    net:add(nn.JoinTable(1, 1))
    net:add(nn.View(heads, args.n_actions))


    if args.gpu >=0 then
        cutorch.setDevice(args.gpu)
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
    end

    return net
	
end