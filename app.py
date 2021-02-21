import os
from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.utils import secure_filename
from PIL import Image
import glob
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'uploads'
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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


model = torch.load("./model.pt", map_location='cpu')


def test(path):
#    from PIL import Image

    img = Image.open(path)
    img = img.resize((32,32))   #resize the image to that of CIFAR 10
    #model = My_neural_net()
    #model.load_state_dict(torch.load("/content/drive/My Drive/model.pkt")).to(device)
#    model = torch.load("./model.pt", )
    
    with torch.no_grad():
        img_tensor = transforms.ToTensor()(img)
        img_tensor = img_tensor.unsqueeze(0)
        # print(img_tensor.shape) #shape = [3, 32, 32]
        model.eval()
        #op=model.conv1(img_tensor)
        #op=op.squeeze(0).numpy()
        #plotting(op)
        
        #from matplotlib import pyplot as plt
        #fig = plt.figure()
        # # # f, axarr = plt.subplots(6,6,figsize=(10,12))
        # # # axarr[0,0].imshow(op[0],cmap='gray')
        # # # plt.figure()
        # plt.imshow(op[0],cmap='gray')
        # plt.show()
        # fig.savefig('/content/drive/My Drive/plot1.png')
        
        
        
        outputs = model(img_tensor)
        p, predicted = torch.max(outputs.data, 1)
        pred_result = classes[predicted[0].item()]
        return {"prediction":pred_result}
       

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)
    if filename != '':
        file_ext = os.path.splitext(filename)[1]
        if file_ext not in app.config['UPLOAD_EXTENSIONS']:
            abort(400)
        uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        path=app.config['UPLOAD_PATH']+'/'+filename
        res=test(path)
        os.remove(path)
        
#        files = glob.glob(app.config['UPLOAD_EXTENSIONS']+'/*')
#        for f in files:
#            os.remove(f)
        return res
# =============================================================================
#         img=Image.open(path)
#         img=img.resize((32,32))
#         with torch.no_grad():
#             img_tensor=transforms.ToTensor()(img)
#             img_tensor=img_tensor.unsqueeze(0)
#             outputs=model(img_tensor)
#             p, predicted=torch.max(outputs.data, 1)
#             pred_result=classes[predicted[0].item()]
#             return {"prediction":pred_result}
# =============================================================================
    return redirect(url_for('index'))

    
if __name__ == '__main__':
    app.run(debug=True)