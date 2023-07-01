import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import pandas as pd
import numpy as np
import seaborn as sns

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels,batch_norm=False):

        super().__init__()

        conv2_params = {'kernel_size': (3, 3),
                        'stride'     : (1, 1),
                        'padding'   : 1
                        }

        noop = lambda x : x

        self._batch_norm = batch_norm

        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=out_channels , **conv2_params)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else noop
        #self.bn1 = nn.GroupNorm(32, out_channels) if batch_norm else noop

        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels, **conv2_params)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else noop
        #self.bn2 = nn.GroupNorm(32, out_channels) if batch_norm else noop

        self.max_pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    @property
    def batch_norm(self):
        return self._batch_norm

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.max_pooling(x)

        return x



class VGG16(nn.Module):

  def __init__(self, input_size, num_classes=11,batch_norm=False):
    super(VGG16, self).__init__()

    self.in_channels,self.in_width,self.in_height = input_size

    self.block_1 = VGGBlock(self.in_channels,64,batch_norm=batch_norm)
    self.block_2 = VGGBlock(64, 128,batch_norm=batch_norm)
    self.block_3 = VGGBlock(128, 256,batch_norm=batch_norm)
    self.block_4 = VGGBlock(256,512,batch_norm=batch_norm)

    self.classifier = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.65),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.65),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )

  @property
  def input_size(self):
      return self.in_channels,self.in_width,self.in_height

  def forward(self, x):

    x = self.block_1(x)
    x = self.block_2(x)
    x = self.block_3(x)
    x = self.block_4(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)

    return x
  
def CelebA_ResNet_classifier(energy_based_model, ds_loaders = [], num_epochs = 20, learning_rate = 0.001, loadpath=''):
    classifier = models.resnet18(pretrained=True)
    num_classes = energy_based_model.Num_classes
    classifier.fc = nn.Linear(classifier.fc.in_features, num_classes)
    if ds_loaders == []: #load old model
        classifier.load_state_dict(torch.load(loadpath)) #DA FARE: AUTOMATIZZA IL LOADPATH
        classifier = classifier.to('cuda')
    else:
        # Definizione del dataloader
        train_loader = ds_loaders[0]
        val_loader = ds_loaders[1]
        # Sblocco di tutti i parametri del modello per il fine-tuning
        for param in classifier.parameters():
            param.requires_grad = True

        # Definizione della loss function e dell'ottimizzatore.
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

        # Spostamento del modello e della loss function sulla GPU, se disponibile
        classifier = classifier.to('cuda')
        criterion = criterion.to('cuda')
        # Ciclo di training
        for epoch in range(num_epochs):
            train_loss = 0.0
            val_loss = 0.0
            correct = 0
            total = 0

            # Training
            classifier.train()
            for images, labels in train_loader:
                images = images.to('cuda')
                labels = labels.to('cuda')
                labels = labels.to(torch.long)  # Convert labels to long data type
                optimizer.zero_grad()
                outputs = classifier(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = correct / total

            # Validation
            classifier.eval()
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to('cuda')
                    labels = labels.to('cuda')
                    labels = labels.to(torch.long)  # Convert labels to long data type
                    outputs = classifier(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                val_loss = val_loss / len(val_loader.dataset)
                val_acc = correct / total

            # Stampa dei risultati dell'epoch
            print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
                .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
            
    return classifier


  
def Classifier_accuracy(input_dict, classifier,model, Thresholding_entropy=[], labels=[], cl_lbls =[], Batch_sz= 100, plot=1, dS=30, l_sz=3):
  classifier = classifier.to('cuda')
  #plot = 2 -> only digitwise accuracy
  if Thresholding_entropy!=[]:
    Thresholding_entropy = torch.mean(Thresholding_entropy) + 2*torch.std(Thresholding_entropy)
  input_data = input_dict['vis_states']
  image_side = int(np.sqrt(input_data.size()[1]))
  #input_data = nr_examples x 784 (i.e. image size) x nr_steps
  
  Cl_pred_matrix = torch.zeros(input_data.size()[0],input_data.size()[2], device=model.DEVICE)
  if not(image_side==64):
    Pred_entropy_mat = torch.zeros(input_data.size()[0],input_data.size()[2], device=model.DEVICE)
    digitwise_avg_entropy = torch.zeros(model.Num_classes,input_data.size()[2], device=model.DEVICE)
    digitwise_sem_entropy = torch.zeros(model.Num_classes,input_data.size()[2], device=model.DEVICE)
  digitwise_acc = torch.zeros(model.Num_classes,input_data.size()[2], device=model.DEVICE)

  acc = torch.zeros(input_data.size()[2])
  if labels==[]:
    labels = torch.zeros(input_data.size()[0], device=model.DEVICE)


  for step in range(input_data.size()[2]):#i.e per ogni step di ricostruzione
    V = input_data[:,:,step] # estraggo i visibili di ciascun esempio allo step iteratore
    V = torch.unsqueeze(V.view((input_data.size()[0],image_side,image_side)),1) #cambio la dimensione del tensore: da n_ex x 784 a n_ex x 1 x 28 x 28
    if image_side==28:
      V_int = F.interpolate(V, size=(32, 32), mode='bicubic', align_corners=False) # creo un nuovo tensore a dimensionalità n_ex x 1 x 32 x 32
    elif image_side==64:
      V_int = F.interpolate(V, size=(224, 224), mode='bicubic', align_corners=False)
      V_int = V_int.repeat(1, 3, 1, 1)
    #tocca fare a batch, il che è abbastanza una fatica. Ma così mangia tutta la GPU
    _dataset = torch.utils.data.TensorDataset(V_int.to('cuda'),labels) # create your datset
    if Batch_sz > input_data.size()[0]: # se il batch size selezionato è più grande dell'input size...
      Batch_sz = input_data.size()[0]
    _dataloader = torch.utils.data.DataLoader(_dataset,batch_size=Batch_sz,drop_last = True) # create your dataloader
    
    index = 0
    acc_v = torch.zeros(math.floor(input_data.size()[0]/Batch_sz))
    #last_batch_size =Batch_sz*acc_v.size()[0] - input_data.size()[0] #per ora non utilizzato
    
    n_it = 0
    for (input, lbls) in _dataloader:
      
      with torch.no_grad():
        pred_vals=classifier(input) #predizioni del classificatore
      if not(image_side==64):
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
      if not(image_side==64):
        digitwise_avg_entropy[digit,step] = torch.mean(Pred_entropy_mat[l[0],step])
        digitwise_sem_entropy[digit,step] = torch.std(Pred_entropy_mat[l[0],step])/math.sqrt(l[0].size()[0])

      inds_digit = Cl_pred_matrix[l[0],step]
      digitwise_acc[digit,step] = torch.sum(inds_digit.to(model.DEVICE)==labels[l[0]])/l[0].size()[0]
  if not(image_side==64):
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
      if cl_lbls==[]:
        cl_lbls = range(model.Num_classes)
      x = range(1,input_data.size()[2]+1)
      if not(image_side==64):
        figure, axis = plt.subplots(2, 2, figsize=(20,15))
        Cl_plot(axis[0,0],x,acc,x_lab='Nr. of steps',y_lab='Classifier accuracy', lim_y = [0,1],Title = 'Classifier accuracy',l_sz=l_sz, dS= dS, color='g')
        Cl_plot(axis[0,1],x,MEAN_entropy,y_err = SEM_entropy,x_lab='Nr. of steps',y_lab='Entropy', lim_y = [0,1],Title = 'Average entropy',l_sz=l_sz, dS= dS, color='r')
        Cl_plot_digitwise(axis[1,0],cl_lbls,x,digitwise_acc,x_lab='Generation step',y_lab='Accuracy',Num_classes=model.Num_classes, lim_y = [0,1],Title = 'Classifier accuracy - digitwise',l_sz=l_sz, dS= dS, cmap=cmap)
        Cl_plot_digitwise(axis[1,1],cl_lbls,x,digitwise_avg_entropy,digitwise_y_err=digitwise_sem_entropy,x_lab='Generation step',y_lab='Entropy',Num_classes=model.Num_classes, lim_y = [0,1],Title = 'Entropy - digitwise',l_sz=l_sz, dS= dS, cmap=cmap)
      else:
        figure, axis = plt.subplots(2, figsize=(10,15))
        Cl_plot(axis[0],x,acc,x_lab='Nr. of steps',y_lab='Classifier accuracy', lim_y = [0,1],Title = 'Classifier accuracy',l_sz=l_sz, dS= dS, color='g')
        Cl_plot_digitwise(axis[1],cl_lbls,x,digitwise_acc,x_lab='Generation step',y_lab='Accuracy',Num_classes=model.Num_classes, lim_y = [0,1],Title = 'Classifier accuracy - digitwise',l_sz=l_sz, dS= dS, cmap=cmap)

      plt.subplots_adjust(left=0.1, 
                        bottom=0.1,  
                        right=0.9,  
                        top=0.9,  
                        wspace=0.4,  
                        hspace=0.4) 
  elif plot==2:
      
      cmap = cm.get_cmap('hsv')
      if cl_lbls==[]:
        cl_lbls = range(model.Num_classes)
      x = range(1,input_data.size()[2]+1)

      figure, axis = plt.subplots(1, 1, figsize=(15,15))
      Cl_plot(axis,x,acc,x_lab='Nr. of steps',y_lab='Classifier accuracy', lim_y = [0,1],Title = 'Classifier accuracy',l_sz=l_sz, dS= dS, color='g')
      figure, axis = plt.subplots(1, 1, figsize=(15,15))
      Cl_plot_digitwise(axis,cl_lbls,x,digitwise_acc,x_lab='Generation step',y_lab='Accuracy', Num_classes=model.Num_classes, lim_y = [0,1],Title = 'Classifier accuracy - digitwise',l_sz=l_sz, dS= dS, cmap=cmap)
      if not(image_side==64):
        figure, axis = plt.subplots(1, 1, figsize=(15,15))      
        Cl_plot(axis,x,MEAN_entropy,y_err = SEM_entropy,x_lab='Nr. of steps',y_lab='Entropy', lim_y = [0,1],Title = 'Average entropy',l_sz=l_sz, dS= dS, color='r')

        figure, axis = plt.subplots(1, 1, figsize=(15,15))
        Cl_plot_digitwise(axis,cl_lbls,x,digitwise_avg_entropy,digitwise_y_err=digitwise_sem_entropy,x_lab='Generation step',y_lab='Entropy', Num_classes=model.Num_classes, lim_y = [0,1],Title = 'Entropy - digitwise',l_sz=l_sz, dS= dS, cmap=cmap)

     
  input_dict['Cl_pred_matrix'] = Cl_pred_matrix
  input_dict['Cl_accuracy'] = acc
  input_dict['digitwise_acc'] = digitwise_acc
  if not(image_side==64):
    input_dict['Pred_entropy_mat'] = Pred_entropy_mat
    input_dict['MEAN_entropy'] = MEAN_entropy
    input_dict['digitwise_entropy'] = digitwise_avg_entropy

   
  return input_dict

def classification_metrics(dict_classifier,model,test_labels=[], Plot=1, dS = 30, rounding=2, T_mat_labels=[], Ian=0):
  '''
  dict_classifier: dizionario del classificatore, con dentro, tra le altre cose, le predizioni del classificatore
  model = la nostra RBM
  test_labels = groundtruth labels
  '''
  if model.Num_classes==10:
    nr_states = model.Num_classes+1 #+1 because we consider also the NON DIGIT class
  else:
    nr_states = model.Num_classes

  Cl_pred_matrix=dict_classifier['Cl_pred_matrix'] #classes predicted by the classifier (nr examples x nr generation steps)
  nr_ex=dict_classifier['Cl_pred_matrix'].size()[0] #number of examples
  #Transition matrix
  Transition_matrix = torch.zeros((nr_states,nr_states))

  for row in Cl_pred_matrix: #for every example classifications
    for nr_genStep in range(1,len(row)): #for each generation step
      Transition_matrix[row[nr_genStep-1].long(),row[nr_genStep].long()] += 1 #add +1 to the entry of the Transition matrix
      #corresponding to the row of the digit of the previous generation step [nr_genStep-1], and the column corresponding to the digit of the 
      #current digit [nr_genStep]
      #IN OTHER WORDS, THE TRANSITION MATRIX IS ESTIMATED FROM ALL THE TRANSITIONS IN THE GENERATED DATASET

  Transition_matrix_rowNorm = Transition_matrix 

  for i in range(nr_states): #for every row of the transition matrix...
    Transition_matrix_rowNorm[i,:] = torch.div(Transition_matrix[i,:],torch.sum(Transition_matrix,1)[i]) #divide each entry by the sum of the entire row (torch.sum(Transition_matrix,1)[i])

  
  #creo la lista delle categorie delle transizioni ad un certo digit

  to_list = [] #this is a string list containing all states (included non digit). They will be the columns of the output dataframes
  for digit in range(nr_states):
    if  digit < model.Num_classes:
      to_list.append(str(digit))
    else:
      to_list.append('Non-digit')


  columns = ['Nr_visited_states','Nr_transitions'] + to_list #the columns of the dataframes created below
  #creo i due dataframes: uno per le medie e l'altro per i relativi errori
  if test_labels == []:
      df_average = pd.DataFrame(columns=columns)
      df_sem = pd.DataFrame(columns=columns)
  else:
      index = range(model.Num_classes) 
      df_average = pd.DataFrame(index=index, columns=columns)
      df_sem = pd.DataFrame(index=index, columns=columns)


  for digit in range(model.Num_classes): #per ogni digit...
    if test_labels!=[]:
      digit_idx = test_labels==digit #trovo la posizione del digit tra le labels groundtruth...
      Vis_digit = dict_classifier['Cl_pred_matrix'][digit_idx,:] #trovo le predizioni del classificatore in quelle posizioni
    else:
      Vis_digit = dict_classifier['Cl_pred_matrix'] #i just take all predictions together
    nr_visited_states_list =[] #qui listerò gli stati che vado a visitare
    nr_transitions_list =[] #qui listerò invece il numero di transizioni che farò
    to_digits_mat = torch.zeros(Vis_digit.size()[0],nr_states) #this is a matrix with nr.rows=nr.examples, nr.cols = 10+1 (nr. digits+no digits category)
    
    for nr_ex,example in enumerate(Vis_digit): # example=tensor of the reconstructions category guessed by the classifier in a single example (i.e. row-wise)
      no10_example = example[example!=10] #here are all generations different from nondigit
      visited_states = torch.unique(no10_example) # find all the states (digits) visited by the RBM (NOTA:NON TENGO CONTO DEL 10(non-digit))
      nr_visited_states = len(visited_states)
      transitions,counts = torch.unique_consecutive(no10_example, return_counts=True) #transitions are the states sequentially explored during the generation
      nr_transitions = len(transitions)
      to_digits = torch.zeros(nr_states) #vector of 11 elements

      transitions,counts = torch.unique_consecutive(example,return_counts=True) #now i include 10 in the transition count
      visited_states = torch.unique(example) #and in the nr of visited states
      # below, for all states visited, i get the nr of steps in which the state was explored
      for state in visited_states:
        idx_state= transitions == state
        to_digits[state.to(torch.long)] = torch.sum(counts[idx_state])

      nr_visited_states_list.append(nr_visited_states)
      nr_transitions_list.append(nr_transitions)
      to_digits_mat[nr_ex,:] = to_digits

    df_average.at[digit,'Nr_visited_states'] = round(sum(nr_visited_states_list)/len(nr_visited_states_list),2)
    df_average.at[digit,'Nr_transitions'] = round(sum(nr_transitions_list)/len(nr_transitions_list),2)
    df_average.iloc[digit,2:] = torch.round(torch.mean(to_digits_mat,0),decimals=2)

    df_sem.at[digit,'Nr_visited_states'] = round(np.std(nr_visited_states_list)/math.sqrt(len(nr_visited_states_list)),2)
    df_sem.at[digit,'Nr_transitions'] = round(np.std(nr_transitions_list)/math.sqrt(len(nr_transitions_list)),2)
    df_sem.iloc[digit,2:] = torch.round(torch.std(to_digits_mat,0)/math.sqrt(to_digits_mat.size()[0]),decimals=2)
  
  #ratio tra passaggi alla classe giusta e la seconda classe di più alta frequenza
  # to_mat = df_average.iloc[:, 2:-1]
  # sem_mat = df_sem.iloc[:, 2:-1]

  if Plot==1:
        
        if test_labels!=[]:
          df_average.plot(y=['Nr_visited_states', 'Nr_transitions'], kind="bar",yerr=df_sem.loc[:, ['Nr_visited_states', 'Nr_transitions']],figsize=(30,10),fontsize=dS)
          plt.xlabel("Class",fontsize=dS)
          if not(model.Num_classes==10):
            plt.xticks(range(model.Num_classes), T_mat_labels, rotation=0)

        else:
          df_average.iloc[0:1].plot(y=['Nr_visited_states', 'Nr_transitions'], kind="bar",yerr=df_sem.loc[:,['Nr_visited_states', 'Nr_transitions']],xticks=[], figsize=(20,10),fontsize=dS)
          
        #plt.title("Classification_metrics-1",fontsize=dS)
        
        plt.ylabel("Number of states",fontsize=dS)
        plt.ylim([0,model.Num_classes])
        plt.legend(["Visited states", "Transitions"], bbox_to_anchor=(0.73,1), loc="upper left", fontsize=dS-dS/3)
        

        #NEW PLOT(PER BRAIN INFORMATICS)
        if Ian ==0:
          lS=25
          Trans_nr = df_average.iloc[:, 2:]
          Trans_nr = Trans_nr.apply(pd.to_numeric)
          Trans_nr = Trans_nr.round(rounding)
          Trans_nr = np.array(Trans_nr)

          Trans_nr_err = df_sem.iloc[:, 2:]
          Trans_nr_err = Trans_nr_err.apply(pd.to_numeric)
          Trans_nr_err = Trans_nr_err.round(rounding)
          Trans_nr_err = np.array(Trans_nr_err)

          StateTimePlot(Trans_nr, T_mat_labels, lS=lS)

          #plot of the transition matrix
          Transition_mat_plot(Transition_matrix_rowNorm,T_mat_labels, lS=lS)
        else:
          
          #OLD PLOT: AVERAGE STATES VISITED BEGINNING FROM A CERTAIN LABEL BIASING DIGIT
          #QUESTO è IL PLOT MOSTRATO IN TESI
          cmap = cm.get_cmap('hsv')
          newcolors = cmap(np.linspace(0, 1, 256))
          black = np.array([0.1, 0.1, 0.1, 1])
          newcolors[-25:, :] = black
          newcmp = ListedColormap(newcolors)
          
          if test_labels!=[]:
            df_average.plot(y=to_list, kind="bar",yerr=df_sem.loc[:, to_list],figsize=(10,10),fontsize=dS,width=0.8,colormap=newcmp)
            plt.xlabel("Digit",fontsize=dS)       
          else:
            df_average.iloc[0:1].plot(y=to_list, kind="bar",yerr=df_sem.loc[:, to_list],figsize=(10,10),fontsize=dS,width=0.8,colormap=newcmp,xticks=[])

          #plt.title("Classification_metrics-2",fontsize=dS)
          
          plt.ylabel("Average nr of steps",fontsize=dS)
          plt.ylim([0,50])
          #plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=dS)

          # Put a legend below current axis
          plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                    fancybox=True, shadow=True, ncol=4, fontsize=dS)
          
           

  return df_average, df_sem, Transition_matrix_rowNorm

def StateTimePlot(Trans_nr, T_mat_labels, lS=25):
        plt.figure(figsize=(15, 15))
        ax = sns.heatmap(Trans_nr, linewidth=0.5, annot=True, annot_kws={"size": lS}, square=True, cbar_kws={"shrink": .82},fmt='.1f', cmap='jet')
        if T_mat_labels==[]:
           T_mat_labels = [str(i) for i in range(len(Trans_nr))]
           ax.set_yticklabels(T_mat_labels)
           if len(Trans_nr) == 10:
              T_mat_labels.append('Non\ndigit')
        ax.set_xticklabels(T_mat_labels)
        ax.tick_params(axis='both', labelsize=lS)

        plt.xlabel('Class', fontsize = 25) # x-axis label with fontsize 15
        plt.ylabel('Biasing Class', fontsize = 25) # y-axis label with fontsize 15
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=lS)
        plt.show()


def Transition_mat_plot(Transition_matrix_rowNorm,T_mat_labels=[], lS=25):
      plt.figure(figsize=(15, 15))
      Transition_matrix=Transition_matrix_rowNorm*100
      ax = sns.heatmap(torch.round(Transition_matrix, decimals=2), linewidth=0.5, annot=True, annot_kws={"size": lS},square=True,cbar_kws={"shrink": .82}, fmt='.1f', cmap='jet')
      plt.xlabel('To', fontsize = 25) # x-axis label with fontsize 15
      plt.ylabel('From', fontsize = 25) # y-axis label with fontsize 15
      if T_mat_labels==[]:
          T_mat_labels = [str(i) for i in range(len(Transition_matrix_rowNorm)-1)]
          if len(Transition_matrix_rowNorm) == 11:
            T_mat_labels.append('Non\ndigit')
      ax.set_xticklabels(T_mat_labels)
      ax.set_yticklabels(T_mat_labels)
      ax.tick_params(axis='both', labelsize=lS)
      cbar = ax.collections[0].colorbar
      cbar.ax.tick_params(labelsize=lS)

      plt.show()
def Cl_plot(axis,x,y,y_err=[],x_lab='Generation step',y_lab='Accuracy', lim_y = [0,1],Title = 'Classifier accuracy',l_sz=3, dS=30, color='g'):
  y=y.cpu()
  
  axis.plot(x, y, c = color, linewidth=l_sz)
  if y_err != []:
    y_err = y_err.cpu()
    axis.fill_between(x,y-y_err, y+y_err, color=color,
                alpha=0.3)
  axis.tick_params(axis='x', labelsize= dS)
  axis.tick_params(axis='y', labelsize= dS)
  axis.set_ylabel(y_lab,fontsize=dS)
  axis.set_ylim(lim_y)
  axis.set_xlabel(x_lab,fontsize=dS)
  axis.set_title(Title,fontsize=dS)

def Cl_plot_digitwise(axis,cl_lbls,x,digitwise_y,digitwise_y_err=[], Num_classes=10,x_lab='Generation step',y_lab='Accuracy', lim_y = [0,1],Title = 'Classifier accuracy - digitwise',l_sz=3, dS= 30, cmap=cm.get_cmap('hsv')):
  c=0
  for digit in range(Num_classes):
    Color = cmap(c/256) 
    MEAN = digitwise_y[digit,:].cpu()
    axis.plot(x, MEAN, c = Color, linewidth=l_sz)
    if digitwise_y_err!=[]:
      SEM = digitwise_y_err[digit,:].cpu()
      axis.fill_between(x,MEAN-SEM, MEAN+SEM, color=Color,
              alpha=0.3)
    c = c+25
  axis.legend(cl_lbls, bbox_to_anchor=(1.04,1), loc="upper left", fontsize=dS) #cambia posizione
  axis.tick_params(axis='x', labelsize= dS)
  axis.tick_params(axis='y', labelsize= dS)
  axis.set_ylabel(y_lab,fontsize=dS)
  axis.set_ylim(lim_y)
  axis.set_xlabel(x_lab,fontsize=dS)
  axis.set_title(Title,fontsize=dS)

def classification_metrics_ZAMBRA(dict_classifier,model,test_labels=[], Plot=1, dS = 30, rounding=2, T_mat_labels=[], Ian=0):
  '''
  dict_classifier: dizionario del classificatore, con dentro, tra le altre cose, le predizioni del classificatore
  model = la nostra RBM
  test_labels = groundtruth labels
  '''
  nr_states = model.Num_classes+1 #+1 because we consider also the NON DIGIT class

  Cl_pred_matrix=dict_classifier['Cl_pred_matrix'] #classes predicted by the classifier (nr examples x nr generation steps)
  nr_ex=dict_classifier['Cl_pred_matrix'].size()[0] #number of examples
  #Transition matrix
  Transition_matrix = torch.zeros((nr_states,nr_states))

  for row in Cl_pred_matrix: #for every example classifications
    for nr_genStep in range(1,len(row)): #for each generation step
      Transition_matrix[row[nr_genStep-1].long(),row[nr_genStep].long()] += 1 #add +1 to the entry of the Transition matrix
      #corresponding to the row of the digit of the previous generation step [nr_genStep-1], and the column corresponding to the digit of the 
      #current digit [nr_genStep]
      #IN OTHER WORDS, THE TRANSITION MATRIX IS ESTIMATED FROM ALL THE TRANSITIONS IN THE GENERATED DATASET

  Transition_matrix_rowNorm = Transition_matrix 

  for i in range(nr_states): #for every row of the transition matrix...
    Transition_matrix_rowNorm[i,:] = torch.div(Transition_matrix[i,:],torch.sum(Transition_matrix,1)[i]) #divide each entry by the sum of the entire row (torch.sum(Transition_matrix,1)[i])

  
  #creo la lista delle categorie delle transizioni ad un certo digit

  to_list = [] #this is a string list containing all states (included non digit). They will be the columns of the output dataframes
  for digit in range(nr_states):
    if  digit < model.Num_classes:
      to_list.append(str(digit))
    else:
      to_list.append('Non-digit')


  columns = ['Nr_visited_states','Nr_transitions'] + to_list #the columns of the dataframes created below
  #creo i due dataframes: uno per le medie e l'altro per i relativi errori
  if test_labels == []:
      df_average = pd.DataFrame(columns=columns)
      df_sem = pd.DataFrame(columns=columns)
  else:
      index = range(model.Num_classes) 
      df_average = pd.DataFrame(index=index, columns=columns)
      df_sem = pd.DataFrame(index=index, columns=columns)


  for digit in range(model.Num_classes): #per ogni digit...
    if test_labels!=[]:
      digit_idx = test_labels==digit #trovo la posizione del digit tra le labels groundtruth...
      Vis_digit = dict_classifier['Cl_pred_matrix'][digit_idx,:] #trovo le predizioni del classificatore in quelle posizioni
    else:
      Vis_digit = dict_classifier['Cl_pred_matrix'] #i just take all predictions together
    nr_visited_states_list =[] #qui listerò gli stati che vado a visitare
    nr_transitions_list =[] #qui listerò invece il numero di transizioni che farò
    to_digits_mat = torch.zeros(Vis_digit.size()[0],model.Num_classes+1) #this is a matrix with nr.rows=nr.examples, nr.cols = 10+1 (nr. digits+no digits category)
    
    for nr_ex,example in enumerate(Vis_digit): # example=tensor of the reconstructions category guessed by the classifier in a single example (i.e. row-wise)
      no10_example = example[example!=10] #here are all generations different from nondigit
      visited_states = torch.unique(no10_example) # find all the states (digits) visited by the RBM (NOTA:NON TENGO CONTO DEL 10(non-digit))
      nr_visited_states = len(visited_states)
      transitions,counts = torch.unique_consecutive(no10_example, return_counts=True) #transitions are the states sequentially explored during the generation
      nr_transitions = len(transitions)
      to_digits = torch.zeros(nr_states) #vector of 11 elements

      transitions,counts = torch.unique_consecutive(example,return_counts=True) #now i include 10 in the transition count
      visited_states = torch.unique(example) #and in the nr of visited states
      # below, for all states visited, i get the nr of steps in which the state was explored
      for state in visited_states:
        idx_state= transitions == state
        to_digits[state.to(torch.long)] = torch.sum(counts[idx_state])

      nr_visited_states_list.append(nr_visited_states)
      nr_transitions_list.append(nr_transitions)
      to_digits_mat[nr_ex,:] = to_digits

    df_average.at[digit,'Nr_visited_states'] = round(sum(nr_visited_states_list)/len(nr_visited_states_list),2)
    df_average.at[digit,'Nr_transitions'] = round(sum(nr_transitions_list)/len(nr_transitions_list),2)
    df_average.iloc[digit,2:] = torch.round(torch.mean(to_digits_mat.cpu(),0),decimals=2)

    df_sem.at[digit,'Nr_visited_states'] = round(np.std(nr_visited_states_list)/math.sqrt(len(nr_visited_states_list)),2)
    df_sem.at[digit,'Nr_transitions'] = round(np.std(nr_transitions_list)/math.sqrt(len(nr_transitions_list)),2)
    df_sem.iloc[digit,2:] = torch.round(torch.std(to_digits_mat.cpu(),0)/math.sqrt(to_digits_mat.size()[0]),decimals=2)
  
  #ratio tra passaggi alla classe giusta e la seconda classe di più alta frequenza
  to_mat = df_average.iloc[:, 2:-1]
  sem_mat = df_sem.iloc[:, 2:-1]

  if Plot==1:
        
        if test_labels!=[]:
          df_average.plot(y=['Nr_visited_states', 'Nr_transitions'], kind="bar",yerr=df_sem.loc[:, ['Nr_visited_states', 'Nr_transitions']],figsize=(30,10),fontsize=dS)
          plt.xlabel("Digit",fontsize=dS)
        else:
          df_average.iloc[0:1].plot(y=['Nr_visited_states', 'Nr_transitions'], kind="bar",yerr=df_sem.loc[:,['Nr_visited_states', 'Nr_transitions']],xticks=[], figsize=(20,10),fontsize=dS)
          
        #plt.title("Classification_metrics-1",fontsize=dS)
        
        plt.ylabel("Number of states",fontsize=dS)
        plt.ylim([0,10])
        plt.legend(["Visited states", "Transitions"], bbox_to_anchor=(0.73,1), loc="upper left", fontsize=dS-dS/3)
        

        #NEW PLOT(PER BRAIN INFORMATICS)
        if Ian ==0:
          lS=25
          Trans_nr = df_average.iloc[:, 2:]
          Trans_nr = Trans_nr.apply(pd.to_numeric)
          Trans_nr = Trans_nr.round(rounding)
          Trans_nr = np.array(Trans_nr)

          Trans_nr_err = df_sem.iloc[:, 2:]
          Trans_nr_err = Trans_nr_err.apply(pd.to_numeric)
          Trans_nr_err = Trans_nr_err.round(rounding)
          Trans_nr_err = np.array(Trans_nr_err)

          StateTimePlot(Trans_nr, Trans_nr_err, T_mat_labels, rounding=1, lS=lS)

          #plot of the transition matrix
          Transition_mat_plot(Transition_matrix_rowNorm.cpu(),T_mat_labels, lS=lS)
        else:
          
          #OLD PLOT: AVERAGE STATES VISITED BEGINNING FROM A CERTAIN LABEL BIASING DIGIT
          #QUESTO è IL PLOT MOSTRATO IN TESI
          cmap = cm.get_cmap('hsv')
          newcolors = cmap(np.linspace(0, 1, 256))
          black = np.array([0.1, 0.1, 0.1, 1])
          newcolors[-25:, :] = black
          newcmp = ListedColormap(newcolors)
          
          if test_labels!=[]:
            df_average.plot(y=to_list, kind="bar",yerr=df_sem.loc[:, to_list],figsize=(10,10),fontsize=dS,width=0.8,colormap=newcmp)
            plt.xlabel("Digit",fontsize=dS)       
          else:
            df_average.iloc[0:1].plot(y=to_list, kind="bar",yerr=df_sem.loc[:, to_list],figsize=(10,10),fontsize=dS,width=0.8,colormap=newcmp,xticks=[])

          #plt.title("Classification_metrics-2",fontsize=dS)
          
          plt.ylabel("Average nr of steps",fontsize=dS)
          plt.ylim([0,50])
          #plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=dS)

          # Put a legend below current axis
          plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                    fancybox=True, shadow=True, ncol=4, fontsize=dS)
          
           

  return df_average, df_sem, Transition_matrix_rowNorm