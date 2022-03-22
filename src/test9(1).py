from torch.autograd import Variable

import numpy as np
import torch
import scipy.io

import torchvision
from torchvision import transforms, datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tqdm

mat = scipy.io.loadmat('monkeydata_training.mat')
trial = mat['trial']
s = []
l = []
bin = 20
for label in range(trial.shape[1]):
    for ind in range(trial.shape[0]):
        spikes = trial[ind][label][1]
        for i in range(int((600-320)/bin)):
            spike = spikes[:,0:320+bin*i]
            sum_spike = np.sum(spike,axis=1)
            s.append(sum_spike)
            l.append(label)


# plt.style.use('ggplot')



# iris = load_iris()
# X = iris['data']
# y = iris['target']
# print(type(y))
X = s
y = np.asarray(l)

# Scale data to have mean 0 and variance 1
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#print(X_scaled)

# Split the data set into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=2)



class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 150)
        self.layer2 = nn.Linear(150, 100)
        self.layer3 = nn.Linear(100, 50)
        self.layer4 = nn.Linear(50, 8)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.softmax(self.layer4(x), dim=1)
        return x

model     = Model(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn   = nn.CrossEntropyLoss()




EPOCHS  = 200
X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test  = Variable(torch.from_numpy(X_test)).float()
y_test  = Variable(torch.from_numpy(y_test)).long()

loss_list     = np.zeros((EPOCHS,))
accuracy_list = np.zeros((EPOCHS,))

for epoch in tqdm.trange(EPOCHS):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss_list[epoch] = loss.item()

    # Zero gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        y_pred = model(X_test)
        #print(y_pred)
        correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
        accuracy_list[epoch] = correct.mean()

print(accuracy_list)
# fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)

# ax1.plot(accuracy_list)
# ax1.set_ylabel("validation accuracy")
# ax2.plot(loss_list)
# ax2.set_ylabel("validation loss")
# ax2.set_xlabel("epochs");

#plt.show()


from pandas import ExcelWriter

import scipy.io as scio


data = {name.replace('.', '_'): parameters.detach().numpy().tolist() for name, parameters in model.named_parameters()}
scio.savemat("weights.mat", data)


# with ExcelWriter('path_to_file.xlsx') as writer:
#     for n, df in enumerate(ddf):
#         df.to_excel(writer,'sheet%s' % n) #save to file
