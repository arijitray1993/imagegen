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
i=0
lr = 1.5
updateInterval = 1
prev_err=0
abs=999

while (i<1000) do
    for j=100,104 do
        -- load dataset
        csv=csvigo.load({path="../../vectors/abstract_v002_train2015_000000000"..j..".csv", verbose=false, mode="raw"})
        csv_tensor=torch.Tensor(csv)
        -- convert to DoubleTensor
        sequence=torch.DoubleTensor(csv_tensor)
        -- define input and target sequence
        input=sequence[{{1,((#sequence)[1] - 1)},{}}]
        target=sequence[{{2,(#sequence)[1]},{}}]

        for s=1,(#input)[1] do
            output= rnn:forward(input[s])
            err = criterion:forward(output, target[s])
            --print(err)
            --if s % 1000 == 0 then
            --    print(err)
            --end
    
            gradOutput = criterion:backward(output, target[s])
            -- the Recurrent layer is memorizing its gradOutputs (up to memSize)
            rnn:backward(input[s], gradOutput)
            if s % rho == 0 then
                rnn:backwardThroughTime()
                rnn:updateParameters(lr)
                rnn:zeroGradParameters()
            end
        end
        
      -- backpropagates through time (BPTT) :
      -- 1. backward through feedback and input layers,
      -- 2. updates parameters
        rnn:backwardThroughTime()
        rnn:updateParameters(lr)
        rnn:zeroGradParameters()
        --rnn:forget()
    end
    rnn:forget()
    i=i+1
    print(err)
end

j=100
csv=csvigo.load({path="../../vectors/abstract_v002_train2015_000000000"..j..".csv", verbose=false, mode="raw"})
csv_tensor=torch.Tensor(csv)
-- convert to DoubleTensor
sequence=torch.DoubleTensor(csv_tensor)
print (sequence[{{2,3},{}}])
--inputtemp=input[1]
for l=1,2 do
out=rnn:forward(sequence[1])
print (out)
inputtemp=out
end
