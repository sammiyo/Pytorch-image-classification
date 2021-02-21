# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:43:30 2020

@author: ysamr
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
device=torch.device('cuda')
class My_neural_net(nn.Module):
    def __init__(self):
        super(My_neural_net, self).__init__()
        #The number of output channels is the number of different kernels used in your ConvLayer. If you would like to output 64 channels, your layer will have 64 different 3x3 kernels
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 5, stride = 1) 
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size = 5, stride = 1) 
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv3 = nn.Conv2d(64, 256, kernel_size = 5, stride = 1) 
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.fc1 = nn.Linear(4*4*256, 2048)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(2048, 80)
        
        self.out = nn.Linear(80, 10)
#         self.fc2 = nn.Linear(120, 10)
#         self.fc3 = nn.Linear(84, 10)
        
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) 
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        # x = self.dropout1(x)
        
        # print('yo',x.shape)
        x = x.view(-1,4*4*256)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
model = My_neural_net().to(device)
loss_func = nn.CrossEntropyLoss()  #define loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)  #define optimizer to be used

    
   

def save_model(model):
    torch.save(model, "./model.pt")


def validate(loss_func, test_loader):    #check the testing accuracy here
    model.eval()
    loss_curr = 0
    true_val = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            output = model(inputs.to(device))  #Gives the probability for each class
            loss = loss_func(output, target.to(device))
            p, predicted = torch.max(output.data, 1)  #the predicted class is the one with the maximum probability
            loss_curr += loss.item()   #.item() gives the numeric value in the tensor
            total += target.size(0)
            t = (predicted.detach().cpu() == target.detach().cpu()).sum().item()
            true_val += t
        return(loss_curr, test_loader, true_val, total)
        
def train(model, train_loader, optimizer, loss_func, test_loader):   
    for epoch in range(10):
        model.train()
        loss_curr = 0.0
        true_val = 0
        total = 0
        for step, (inputs, target) in enumerate(train_loader):
              model.train()   # set the model in training mode
              optimizer.zero_grad()
              output = model(inputs.to(device))
              #print("1")
              loss = loss_func(output, target.to(device))
              #print("2")
              loss.backward()
              optimizer.step()
              #print("3")
              p, predicted = torch.max(output.data, 1)
              #print("4")
              loss_curr += loss.item()   #.item() gives the numeric value in the tensor 
              total += target.size(0)
              t = (predicted.detach().cpu() == target.detach().cpu()).sum().item()
              true_val += t                
                            
        l, tl, tv, t = validate(loss_func, test_loader)
        print('Epoch: %d/10, Train loss: %.3f, Train Accuracy: %.2f, Test loss: %.3f, Test Accuracy: %.2f' %(epoch + 1, loss_curr / len(train_loader), 
                                                                                        (true_val*100) / total, l / len(tl), (tv*100)/ t))
    save_model(model)
    return model
        


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def plotting(op):
    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(6,6,figsize=(10,12))
    plt.axis('off')
    axarr[0,0].imshow(op[0],cmap='gray')
    axarr[0,1].imshow(op[1],cmap='gray')
    axarr[0,2].imshow(op[2],cmap='gray')
    axarr[0,3].imshow(op[3],cmap='gray')
    axarr[0,4].imshow(op[4],cmap='gray')
    axarr[0,5].imshow(op[5],cmap='gray')
    axarr[1,0].imshow(op[6],cmap='gray')
    axarr[1,1].imshow(op[7],cmap='gray')
    axarr[1,2].imshow(op[8],cmap='gray')
    axarr[1,3].imshow(op[9],cmap='gray')
    axarr[1,4].imshow(op[10],cmap='gray')
    axarr[1,5].imshow(op[11],cmap='gray')
    axarr[2,0].imshow(op[12],cmap='gray')
    axarr[2,1].imshow(op[13],cmap='gray')
    axarr[2,2].imshow(op[14],cmap='gray')
    axarr[2,3].imshow(op[15],cmap='gray')
    axarr[2,4].imshow(op[16],cmap='gray')
    axarr[2,5].imshow(op[17],cmap='gray')
    axarr[3,0].imshow(op[18],cmap='gray')
    axarr[3,1].imshow(op[19],cmap='gray')
    axarr[3,2].imshow(op[20],cmap='gray')
    axarr[3,3].imshow(op[21],cmap='gray')
    axarr[3,4].imshow(op[22],cmap='gray')
    axarr[3,5].imshow(op[23],cmap='gray')
    axarr[4,0].imshow(op[24],cmap='gray')
    axarr[4,1].imshow(op[25],cmap='gray')
    axarr[4,2].imshow(op[26],cmap='gray')
    axarr[4,3].imshow(op[27],cmap='gray')
    axarr[4,4].imshow(op[28],cmap='gray')
    axarr[4,5].imshow(op[29],cmap='gray')
    axarr[5,0].imshow(op[30],cmap='gray')
    axarr[5,1].imshow(op[31],cmap='gray')
    f.savefig('/content/drive/My Drive/CONV_rslt.png')
    
    
#Predict the class using the images
def test(path):
    from PIL import Image

    img = Image.open(path)
    img = img.resize((32,32))   #resize the image to that of CIFAR 10
    #model = My_neural_net()
    #model.load_state_dict(torch.load("/content/drive/My Drive/model.pkt")).to(device)
    model = torch.load("./model.pt").to(device)
    
    with torch.no_grad():
        img_tensor = transforms.ToTensor()(img)
        img_tensor = img_tensor.unsqueeze(0)
        # print(img_tensor.shape) #shape = [3, 32, 32]
        model.eval()
        op=model.conv1(img_tensor.to(device))
        op=op.squeeze(0).detach().cpu().numpy()
        plotting(op)
        
        #from matplotlib import pyplot as plt
        #fig = plt.figure()
        # # # f, axarr = plt.subplots(6,6,figsize=(10,12))
        # # # axarr[0,0].imshow(op[0],cmap='gray')
        # # # plt.figure()
        # plt.imshow(op[0],cmap='gray')
        # plt.show()
        # fig.savefig('/content/drive/My Drive/plot1.png')
        
        
        
        outputs = model(img_tensor.to(device))
        p, predicted = torch.max(outputs.data, 1)
        pred_result = classes[predicted[0].item()]
        print("Prediction Result : {}".format(pred_result))




import sys
if __name__ == '__main__':
    train_data = torchvision.datasets.CIFAR10(root='./data.cifar10', train=True, transform=torchvision.transforms.ToTensor(), 
                                              download=True)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=2)
    
    test_data = torchvision.datasets.CIFAR10(root='./data.cifar10/', train=False, transform=torchvision.transforms.ToTensor())
    test_loader = Data.DataLoader(dataset=test_data, batch_size=4, shuffle=False, num_workers=2)
    
    
    if len(sys.argv)==2:
        try:
            assert sys.argv[1] == "train"
            model = train(model, train_loader, optimizer, loss_func, test_loader)
        except AssertionError:
            print('Enter either train or test')

    elif len(sys.argv)==3:
        test(sys.argv[2])
       
   
