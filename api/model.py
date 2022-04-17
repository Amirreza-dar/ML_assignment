import os
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import requests
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from pathlib import Path


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    


def fet_data():
    import cv2
    X_chess = list()
    Y_chess = list()
    mother_folder = '../data/seg_train'
    classes = os.listdir(mother_folder)
    for folder in classes:
        #  print(folder)
         i = 0
         for file in os.listdir(mother_folder + '/' + folder):
            if i > 800:
                break
            img = cv2.imread(mother_folder + '/' + folder + '/' + file)
            try:
                    img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
                    X_chess.append(img.T)
                    Y_chess.append(folder)
            except:
                    pass
            i += 1

    X = np.array(X_chess)
    y = np.array(Y_chess)

    from sklearn.preprocessing import LabelEncoder
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y)

    from torch.utils.data import TensorDataset
    from sklearn.model_selection import train_test_split

    x_tensor = torch.from_numpy(np.array(X)).float()
    Yt_train = torch.from_numpy(np.array(y)).int()
    y_tensor = Yt_train.type(torch.LongTensor)

    data = dict()
    for ind,label in enumerate(y_encoder.inverse_transform([0,1,2,3,4,5])):
        data[ind] = label
    
    try:
        import cPickle as pickle
    except ImportError:  # Python 3.x
        import pickle


    with open('labels.p', 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

    train_input, test_input, train_target, test_target = train_test_split(x_tensor, y_tensor, test_size=0.1, random_state=431)

    batch_size = 16

    train_dataset =TensorDataset(train_input, train_target)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size,
                                              shuffle=True, num_workers=2)


    return trainloader


class Net(nn.Module):
    def __init__(self):
        super().__init__()        
        self.conv1 = nn.Conv2d(3, 6, 9)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(6, 16, 7)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 256)
        self.fc3 = nn.Linear(256, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists("cifar_net.pth"):
        start_time = time.time()
        status = False

        trainloader = fet_data()
    
        net = Net().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(15):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        train_time = time.time() - start_time
        PATH = 'cifar_net.pth'
        torch.save(net, PATH)
        status = True
        return status
    else:
        return True



def predict(image_link):

    import time
    from sklearn.preprocessing import LabelEncoder
    import pickle

    
    file_id = image_link.split('/')[-2]
    destination = 'image.png'
    download_file_from_google_drive(id, destination)


    download_file_from_google_drive(file_id, destination)
    image = cv2.imread('image.png')
    image = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA).T
    image = np.array(image)
    image = image[None,:,:,:]
    start_time = time.time()
    image = torch.from_numpy(image).float()
    try:
        with torch.no_grad():
            net = torch.load('cifar_net.pth')
            net.eval()
            # print('eyval')        
    except:
        return 'data has not been trained!!'

    with open('labels.p', 'rb') as fp:
        data = pickle.load(fp)

    output = net(image)

    label = data[np.argmax(output.detach().numpy())]
    output.data
    prediction_time = time.time() - start_time

    return  label,prediction_time



if __name__ == '__main__':
    status = train()
    img_link = 'https://drive.google.com/file/d/1CDZLGWf6oRWAnHolm2bnhdJRRPxSscvd/view?usp=sharing'
    print(predict(img_link))


