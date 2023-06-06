
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
import VGG_MNIST
from VGG_MNIST import *
import Study_generativity
from Study_generativity import *



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        if not(num_classes==40):
          self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        if not(self.num_classes==40):
          out = self.softmax(out)

        return out


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if model.num_classes==40:
          loss = criterion(output, target.float())
          predicted = torch.sigmoid(output) >= 0.5
        else:
          loss = criterion(output, target.long())
          predicted = torch.argmax(output, axis = 1)
        # Aggiungi regolarizzazione L2 ai pesi della rete
        l2_reg = torch.tensor(0.)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += 0.001 * l2_reg
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total += torch.numel(target)
        correct += (predicted == target).sum().item()

    train_loss /= len(train_loader.dataset)
    accuracy = correct / total
    return train_loss, accuracy


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if model.num_classes==40:
              loss = criterion(output, target.float())
              predicted = torch.sigmoid(output) >= 0.5
            else:
              loss = criterion(output, target.long())
              predicted = torch.argmax(output, axis = 1)
            total += torch.numel(target)
            correct += (predicted == target).sum().item()
            test_loss += loss.item()
    test_loss /= len(test_loader.dataset)
    accuracy = correct / total
    return test_loss, accuracy


def main(train_loader, test_loader, num_classes=40):
    # Hyperparameters
    learning_rate = 0.001
    batch_size = 64
    epochs = 10

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model initialization
    model = ResNet(num_classes=num_classes).to(device)

    # Utilizza la funzione di perdita BCEWithLogitsLoss
    if num_classes==40:
      criterion = nn.BCEWithLogitsLoss()
    else:
      criterion =nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Utilizza un learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2)

    # Training and evaluation
    for epoch in range(1, epochs+1):

        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = test(model,device, test_loader, criterion)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        scheduler.step(test_loss)

    return model


def Classifier_accuracy(input_dict, Classifier,model, Thresholding_entropy=[], labels=[], Batch_sz= 80, plot=1, dS=30, l_sz=3):

  #plot = 2 -> only digitwise accuracy
  if Thresholding_entropy!=[]:
    Thresholding_entropy = torch.mean(Thresholding_entropy) + 2*torch.std(Thresholding_entropy)
  input_data = input_dict['vis_states']
  image_side = int(np.sqrt(input_data.size()[1]))
  #input_data = nr_examples x vectorized image size x nr_steps
  
  Cl_pred_matrix = torch.zeros(input_data.size()[0],input_data.size()[2], device=model.DEVICE)
  Pred_entropy_mat = torch.zeros(input_data.size()[0],input_data.size()[2], device=model.DEVICE)
  digitwise_acc = torch.zeros(model.Num_classes,input_data.size()[2], device=model.DEVICE)
  digitwise_avg_entropy = torch.zeros(model.Num_classes,input_data.size()[2], device=model.DEVICE)
  digitwise_sem_entropy = torch.zeros(model.Num_classes,input_data.size()[2], device=model.DEVICE)
  acc = torch.zeros(input_data.size()[2])
  if labels==[]:
    labels = torch.zeros(input_data.size()[0], device=model.DEVICE)


  for step in range(input_data.size()[2]):#i.e per ogni step di ricostruzione
    V = input_data[:,:,step] # estraggo i visibili di ciascun esempio allo step iteratore
    V = torch.unsqueeze(V.view((input_data.size()[0],image_side,image_side)),1) #cambio la dimensione del tensore: da n_ex x 784 a n_ex x 1 x 28 x 28
    #V_int = F.interpolate(V, size=(32, 32), mode='bicubic', align_corners=False) # creo un nuovo tensore a dimensionalità n_ex x 1 x 32 x 32
    #tocca fare a batch, il che è abbastanza una fatica. Ma così mangia tutta la GPU
    _dataset = torch.utils.data.TensorDataset(V,labels) # create your datset
    if Batch_sz > input_data.size()[0]: # se il batch size selezionato è più grande dell'input size...
      Batch_sz = input_data.size()[0]
    _dataloader = torch.utils.data.DataLoader(_dataset,batch_size=Batch_sz,drop_last = True) # create your dataloader
    
    index = 0
    acc_v = torch.zeros(math.floor(input_data.size()[0]/Batch_sz))
    #last_batch_size =Batch_sz*acc_v.size()[0] - input_data.size()[0] #per ora non utilizzato
    
    n_it = 0
    for (input, lbls) in _dataloader:
      
      with torch.no_grad():
        pred_vals=Classifier(input) #predizioni del classificatore
      
      Pred_entropy = torch.distributions.Categorical(probs =pred_vals[:,:10]).entropy()
      Pred_entropy_mat[index:index+Batch_sz,step] = Pred_entropy

      _, inds = torch.max(pred_vals,dim=1)
      Cl_pred_matrix[index:index+Batch_sz,step] = inds
      acc_v[n_it] = torch.sum(inds.to(model.DEVICE)==lbls)/input.size()[0]

      n_it = n_it+1
      index = index+ Batch_sz
    acc[step] = torch.mean(acc_v)
    
    for digit in range(model.Num_classes):
      l = torch.where(labels == digit)

      
      digitwise_avg_entropy[digit,step] = torch.mean(Pred_entropy_mat[l[0],step])
      digitwise_sem_entropy[digit,step] = torch.std(Pred_entropy_mat[l[0],step])/math.sqrt(l[0].size()[0])

      inds_digit = Cl_pred_matrix[l[0],step]
      digitwise_acc[digit,step] = torch.sum(inds_digit.to(model.DEVICE)==labels[l[0]])/l[0].size()[0]

  MEAN_entropy = torch.mean(Pred_entropy_mat,0)
  SEM_entropy = torch.std(Pred_entropy_mat,0)/math.sqrt(input_data.size()[0])

  if Thresholding_entropy!=[]:
    #  Entropy_mat_NN = Pred_entropy_mat[Cl_pred_matrix==10]
    #  NN_mean_entropy = Entropy_mat_NN.mean()
    #  NN_std_entropy = Entropy_mat_NN.std()
     Cl_pred_matrix[Pred_entropy_mat>=Thresholding_entropy]=10

     Lab_mat= labels.unsqueeze(1).expand(len(labels), input_data.size()[2])

     Cl_acc = Cl_pred_matrix==Lab_mat
     Cl_acc = Cl_acc.to(torch.float)
     acc =Cl_acc.mean(dim=0)
     
     for digit in range(model.Num_classes):
        digit_idxs = labels==digit
        a = Cl_pred_matrix[digit_idxs,:]==Lab_mat[digit_idxs,:]
        a = a.to(torch.float)
        digitwise_acc[digit,:]=a.mean(dim=0)     


  if plot == 1:
      c=0
      cmap = cm.get_cmap('hsv')
      lbls = range(model.Num_classes)
      x = range(1,input_data.size()[2]+1)

      figure, axis = plt.subplots(2, 2, figsize=(20,15))
      Cl_plot(axis[0,0],x,acc,x_lab='Nr. of steps',y_lab='Classifier accuracy', lim_y = [0,1],Title = 'Classifier accuracy',l_sz=l_sz, dS= dS, color='g')
      Cl_plot(axis[0,1],x,MEAN_entropy,y_err = SEM_entropy,x_lab='Nr. of steps',y_lab='Entropy', lim_y = [0,1],Title = 'Average entropy',l_sz=l_sz, dS= dS, color='r')
      Cl_plot_digitwise(axis[1,0],lbls,x,digitwise_acc,x_lab='Generation step',y_lab='Accuracy', lim_y = [0,1],Title = 'Classifier accuracy - digitwise',l_sz=l_sz, dS= dS, cmap=cmap, Num_classes=model.Num_classes)
      Cl_plot_digitwise(axis[1,1],lbls,x,digitwise_avg_entropy,digitwise_y_err=digitwise_sem_entropy,x_lab='Generation step',y_lab='Entropy', lim_y = [0,1],Title = 'Entropy - digitwise',l_sz=l_sz, dS= dS, cmap=cmap, Num_classes=model.Num_classes)

      plt.subplots_adjust(left=0.1, 
                        bottom=0.1,  
                        right=0.9,  
                        top=0.9,  
                        wspace=0.4,  
                        hspace=0.4) 
  elif plot==2:
      
      cmap = cm.get_cmap('hsv')
      lbls = range(model.Num_classes)
      x = range(1,input_data.size()[2]+1)
      figure, axis = plt.subplots(1, 1, figsize=(15,15))

      Cl_plot(axis,x,acc,x_lab='Nr. of steps',y_lab='Classifier accuracy', lim_y = [0,1],Title = 'Classifier accuracy',l_sz=l_sz, dS= dS, color='g')
      figure, axis = plt.subplots(1, 1, figsize=(15,15))      
      Cl_plot(axis,x,MEAN_entropy,y_err = SEM_entropy,x_lab='Nr. of steps',y_lab='Entropy', lim_y = [0,1],Title = 'Average entropy',l_sz=l_sz, dS= dS, color='r')
      figure, axis = plt.subplots(1, 1, figsize=(15,15))
      Cl_plot_digitwise(axis,lbls,x,digitwise_acc,x_lab='Generation step',y_lab='Accuracy', lim_y = [0,1],Title = 'Classifier accuracy - digitwise',l_sz=l_sz, dS= dS, cmap=cmap)
      figure, axis = plt.subplots(1, 1, figsize=(15,15))
      Cl_plot_digitwise(axis,lbls,x,digitwise_avg_entropy,digitwise_y_err=digitwise_sem_entropy,x_lab='Generation step',y_lab='Entropy', lim_y = [0,1],Title = 'Entropy - digitwise',l_sz=l_sz, dS= dS, cmap=cmap)

     


  input_dict['Cl_pred_matrix'] = Cl_pred_matrix
  input_dict['Cl_accuracy'] = acc
  input_dict['digitwise_acc'] = digitwise_acc
  input_dict['Pred_entropy_mat'] = Pred_entropy_mat
  input_dict['MEAN_entropy'] = MEAN_entropy
  input_dict['digitwise_entropy'] = digitwise_avg_entropy

   
  return input_dict