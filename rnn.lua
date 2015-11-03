require 'rnn'
require 'csvigo'

--batchSize = 8
rho = 4
hiddenSize = 150
inputSize = 191
-- RNN
r = nn.Recurrent(
   hiddenSize, nn.Linear(inputSize, hiddenSize), 
   nn.Linear(hiddenSize, hiddenSize), nn.ReLU(), 
   rho
)

rnn = nn.Sequential()
--rnn:add(nn.Linear(inputSize, inputSize))
--rnn:add(nn.Tanh())
rnn:add(r)
rnn:add(nn.Linear(hiddenSize, inputSize))
rnn:add(nn.ReLU())
rnn:add(nn.Linear(inputSize, inputSize))
rnn:add(nn.ReLU())

criterion = nn.AbsCriterion()

-- load dataset

--csv=csvigo.load({path="../../vectors/abstract_v002_train2015_00000000000"..j..".csv", verbose=false, mode="raw"})
csv=csvigo.load({path="trainvectors.csv", verbose=false, mode="raw"})
csv_tensor=torch.Tensor(csv)
sequence=torch.DoubleTensor(csv_tensor)

-- dummy dataset (task is to predict next item, given previous)
--sequence={{1,2,3,4},{2,4,6,8},{4,8,12,16},{8,16,24,32},{4,8,12,16},{2,4,6,8},{1,2,3,4},{2,4,6,8},{4,8,12,16}}

--sequence=torch.DoubleTensor(sequence)
--for c=1,5 do
input=sequence[{{1,1000},{}}]
target=sequence[{{2,1001},{}}]
--c=c+6
lr = 1.5
updateInterval = 1
i = 1
prev_err=0
abs=999
while (i<40000 and abs>0) do
   -- a batch of inputs
   local output = rnn:forward(input)
   -- incement indices
   local err = criterion:forward(output, target)
   print(err)
   local gradOutput = criterion:backward(output, target)
   -- the Recurrent layer is memorizing its gradOutputs (up to memSize)
   rnn:backward(input, gradOutput)

   i = i + 1
   -- note that updateInterval < rho
   if i % updateInterval == 0 then
      -- backpropagates through time (BPTT) :
      -- 1. backward through feedback and input layers,
      -- 2. updates parameters
      rnn:backwardThroughTime()
      rnn:updateParameters(lr)
      rnn:zeroGradParameters()
   end
   abs=(err-prev_err)^2
   prev_err=err
end
--print(prev_err)
--end
print (rnn:forward(input))
print (target)
torch.save('rnn1',rnn)
