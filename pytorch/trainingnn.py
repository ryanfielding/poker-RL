import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

# define data
# this would be states in poker game
# such as pot size, stack sizes, hole cards, etc.

# build the network
model = nn.Sequential(nn.Linear(64, 10),
                    nn.ReLU(), 
                    nn.Linear(10, 3), 
                    nn.LogSoftmax(dim=1))

#define loss and optimizer
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)



#training
epochs = 5 #for 100 games of poker for example
for e in range(epochs):
    running_loss = 0
    #get updated states of game for inputs
    states = 1 #this should feed in states from the poker engine
    #reset gradients
    optimizer.zero_grad()
    #fwd pass to decide agent action
    action = model.forward(states)
    #calculate losses or gains to backpropogate
    loss = criterion(action, 10) #10 should be 'target'
    loss.backward()
    #update weights
    optimizer.step()
    running_loss += loss.item()

#save model once trained
import helper
#should use this if loading into model of diff architecture
# modelV1checkpoint = {'input_size': 64,
#                     'output_size': 10,
#                     'hidden_layers': [each.out_features for each in model.hidden_layers],
#                     'state_dict': model.state_dict()}
torch.save(model.state_dict(), 'modelV1.pth')
#can re load model with tuned weights
state_dict = torch.load('modelV1.pth')
model.load_state_dict(state_dict)
