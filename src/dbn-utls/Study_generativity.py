from tqdm import tqdm #tqdm is a Python library that provides a way to create progress bars for loops and iterators, 
#making it easier to track the progress of lengthy operations.
import os
import json
import numpy as np
from torchvision import datasets,transforms
from torch.utils.data import Dataset, DataLoader
import torch
from dbns import *
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import Classifiers
import methods
from Classifiers import *
from methods import *
from google.colab import files
from itertools import combinations
from skimage.filters import threshold_sauvola
from pathlib import Path
from copy import deepcopy

def data_and_labels(data_train, BATCH_SIZE,NUM_FEAT,DATASET_ID,n_cols_labels):
  #This function prepares the data for processing by machine learning models (NOTE FOR THE WRITER: Be more specific)
  train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True) #create a dataloader with the data shuffled
  num_batches = data_train.__len__() // BATCH_SIZE # Calculate the total number of batches (NOTE: data_train.__len__() is equivalent to len(data_train))
  # Create empty tensors to store the training data and labels
  train_data = torch.empty(num_batches, BATCH_SIZE, NUM_FEAT)
  train_labels = torch.empty(num_batches, BATCH_SIZE, n_cols_labels)
  with tqdm(train_loader, unit = 'Batch') as tdata: #unit='Batch': Specifies the unit of measurement displayed by the progress bar
      #Inside the with block, you typically have a loop that iterates over train_loader, 
      #and tqdm will automatically update and display the progress bar as the loop progresses.
      for idx, (batch, labels) in enumerate(tdata):
          tdata.set_description(f'Train Batch {idx}\t') # set a description for the progress bar.
          if idx<num_batches: # Check if the current batch index is within the number of batches
            bsize = batch.shape[0] # Get the batch size of the current batch
            if DATASET_ID =='MNIST':
              train_data[idx,:,:] = batch.reshape(bsize, -1).type(torch.float32) #reshape images into vectors and change the type of the elements to torch.float32
            else:
              gray_batch = batch.mean(dim=1, keepdim=True) # convert to grayscale
              for i in range(bsize): #for every element in the batch...
                  img_to_store = gray_batch[i].numpy().squeeze() #convert the image to numpy and eliminate dimensions = 1
                  if 'BW' in DATASET_ID: #if the DATASET_ID includes the BW letters (that stand for 'Black and White')...
                    #...apply the Sauvola-Pietikainen algorithm for binarization. Parameters follow the paper https://link.aps.org/doi/10.1103/PhysRevX.13.021003
                    threshold = threshold_sauvola(img_to_store,window_size=7, k=0.05) 
                    img_to_store = img_to_store > threshold

                  train_data[idx, i, :] = torch.from_numpy(img_to_store.reshape(-1).astype(np.float32)) #convert the np array into torch tensor (as a 1d vector (-> reshape))
            if len(labels.shape)==1: #if your labels have just 1 dimension...
                labels = labels.unsqueeze(1) #... then add 1 dimension
            train_labels[idx, :, :] = labels.type(torch.float32) #store also labels as torch.float32
  return train_data, train_labels  


def load_data_ZAMBRA(CPARAMS,LPARAMS,Zambra_folder_drive):
    #This function summarizes the loading data into the Zambra repository, avoiding issues and errors.
    DATASET_ID = CPARAMS['DATASET_ID']
    n_cols_labels = 1 # Initialize the number of label columns (e.g. 1 for MNIST)
    # Determine the number of features based on DATASET_ID
    if DATASET_ID =='MNIST':
       NUM_FEAT= np.int32(28*28)
    elif DATASET_ID =='CIFAR10':
      NUM_FEAT= np.int32(32*32)
    elif 'CelebA' in DATASET_ID:
       NUM_FEAT= np.int32(64*64)
       n_cols_labels = 40

    BATCH_SIZE = LPARAMS['BATCH_SIZE']
    # Create names for test and training data files
    test_filename = 'test_dataset_'+DATASET_ID+'.npz'
    train_filename = 'train_dataset_'+DATASET_ID+'.npz'
    trainfile_path= os.path.join(Zambra_folder_drive,'dataset_dicts',train_filename)
    testfile_path = os.path.join(Zambra_folder_drive,'dataset_dicts',test_filename)

    # If the training file exists, load the data (both train and test)
    if os.path.exists(trainfile_path):
      train_dataset = dict(np.load(trainfile_path))
      test_dataset = dict(np.load(testfile_path))
      # Convert the numpy arrays to torch tensors
      for key in train_dataset:
          train_dataset[key] = torch.from_numpy(train_dataset[key])
          test_dataset[key]= torch.from_numpy(test_dataset[key])

      Real_BATCH_SIZE = train_dataset['data'].shape[1] # Calculate the actual BATCH_SIZE of the training data
      if not(Real_BATCH_SIZE ==BATCH_SIZE): #if there is a mismatch between the actual batch size and the desired batch size (i.e. BATCH_SIZE)...
        #...then ask the user if he wants to reshape the data to the desired batch size
        reshape_yn = int(input('data found with batchsize '+str(Real_BATCH_SIZE)+ '.Reshape it to the desired batchsize '+str(BATCH_SIZE)+'? (1=y,0=n)'))
        if reshape_yn==1: # if the user asks for reshape...
          def reshape_data(train_dataset, Real_BATCH_SIZE, BATCH_SIZE):
              n = BATCH_SIZE//Real_BATCH_SIZE # Designed for 128 and 64 only
              if not(train_dataset['data'].shape[0]%n==0): #if the nr of rows of train_dataset['data'] is not divisible by n...
                  #remove a row from the data and label arrays
                  train_dataset['data'] = train_dataset['data'][:-1,:,:]
                  train_dataset['labels'] = train_dataset['labels'][:-1,:,:]
              #reshape both the data and labels array so that the batch size is now the desired one (i.e. BATCH_SIZE)
              train_dataset['data'] = train_dataset['data'].view(train_dataset['data'].shape[0]//n, BATCH_SIZE, train_dataset['data'].shape[2])
              train_dataset['labels'] = train_dataset['labels'].view(train_dataset['labels'].shape[0]//n, BATCH_SIZE, train_dataset['labels'].shape[2])
              return train_dataset
          #...reshape both the train and test dataset
          train_dataset = reshape_data(train_dataset, Real_BATCH_SIZE, BATCH_SIZE)
          test_dataset = reshape_data(test_dataset, Real_BATCH_SIZE, BATCH_SIZE)
    else: # If the training file does not exist, load the data from scratch based on DATASET_ID
      if DATASET_ID =='MNIST':
        transform =transforms.Compose([transforms.ToTensor()])
        data_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
        data_test = datasets.MNIST('../data', train=False, download=True, transform=transform)

      elif DATASET_ID=='CIFAR10':
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()])
        data_train = datasets.CIFAR10(root='../data', split='train', download=True, transform=transform)
        data_test = datasets.CIFAR10(root='../data', split='test', download=True, transform=transform)   
        
      elif 'CelebA' in DATASET_ID:
        transform=transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(),
            transforms.ToTensor()])
        data_train = datasets.CelebA(root='../data', split='train', download=True, transform=transform)
        data_test = datasets.CelebA(root='../data', split='test', download=True, transform=transform)      
      train_data, train_labels = data_and_labels(data_train, BATCH_SIZE,NUM_FEAT,DATASET_ID,n_cols_labels)
      test_data, test_labels = data_and_labels(data_test, BATCH_SIZE,NUM_FEAT,DATASET_ID,n_cols_labels)

      train_dataset = {'data': train_data, 'labels': train_labels}
      test_dataset = {'data': test_data, 'labels': test_labels}
      Path(os.path.join(Zambra_folder_drive,'dataset_dicts')).mkdir(exist_ok=True)  #If the directory already exists, it does nothing (thanks to "exist_ok=True")
      #This is a common way to save multiple arrays or data structures into a single archive file for easy storage and later retrieval.
      #'**' unpacks the dictionary and passes its contents to the function.
      np.savez(trainfile_path, **train_dataset)
      np.savez(testfile_path, **test_dataset)

    return train_dataset, test_dataset

class MyDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        # Initialize the dataset with features, labels, and an optional transform function
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Define the behavior when len() is called on an instance of MyDataset
        return len(self.features)

    def __getitem__(self, idx):
        # Define the behavior when an item is accessed using indexing
        feature = self.features[idx]
        label = self.labels[idx]

        # apply transformations if specified
        if self.transform:
            feature = self.transform(feature)

        return feature, label

def Multiclass_dataset(train_dataset, selected_idx = [20,31], for_classifier = False, Old_rbm=False, DEVICE ='cuda'):
  Batch_size = train_dataset['data'].shape[1]
  Train_data = copy.deepcopy(train_dataset['data']).to(DEVICE) # Deep copy and move training data to the specified device (e.g., 'cuda')
  if not(selected_idx==[]): # If selected indices are provided, deep copy and move the corresponding labels to the device
    Train_labels = copy.deepcopy(train_dataset['labels'][:,:,selected_idx]).to(DEVICE)
  else:
    Train_labels = copy.deepcopy(train_dataset['labels']).to(DEVICE)
  side_img = int(np.sqrt(Train_data.shape[2])) # Calculate the side length of the image (NOTE: the images are squares, and are originally stored as vectors)
  Train_data = Train_data.view(Train_data.shape[0]*Train_data.shape[1], side_img, side_img).unsqueeze(1) #here i am storing the data without batching them, and as images (i.e. matrices instead of vectors)
  Train_labels = Train_labels.view(Train_labels.shape[0]*Train_labels.shape[1], Train_labels.shape[2]) # i store also the labels without batching


  if not(selected_idx==[]): # If selected indices are provided
    #Here i transform multilabels (e.g. blonde and with sunglasses) into one-hot encoded labels, like in the MNIST. This part of the code is used for CelebA labels in particular
    powers_of_10 = torch.pow(10, torch.arange(len(selected_idx), dtype=torch.float)).to(DEVICE) # this generates a tensor of powers of 10, where each element corresponds to a different power of 10
    Cat_labels = torch.matmul(Train_labels,powers_of_10) #for each multilabel (e.g. 1 0 1 1) i compute the matmul with powers_of_10, so to obtain a single number (e.g. 1101)
    # now in Cat_labels each each element is labelled with a single category, like in MNIST
    cats, f_cat = torch.unique(Cat_labels, return_counts= True) #i compute the frequency of each of the categories in Cat_labels
    lowest_freq_idx = torch.argmin(f_cat)#i find the index of the lower frequency category...
    lowest_freq = f_cat[lowest_freq_idx]#...and using it the frequency of that category...
    lowest_freq_cat = cats[lowest_freq_idx]#...and its identity (lowest_freq_cat)

    for category in cats: #for every category present in Cat_labels...
      if not(category==lowest_freq_cat): #if that category is not the one with the lowest frequency...
        cat_freq = f_cat[cats==category] #find the frequency of that category
        cat_indexes = torch.where(Cat_labels == category)[0] #...and the indexes of elements identified by that category
        #below i select a random subpopulation of 'category' (of size = to the frequency difference with the lowest freq category) that i will later delete from the dataset
        #in order to have all the labels balanced (i.e. all with the same number of elements)
        if category==0: #if the category is the first one to be iterated (i.e. 0)
          indexes_to_delete = cat_indexes[torch.randperm(len(cat_indexes))[:cat_freq-lowest_freq]]
        else:
          new_indexes_to_delete = cat_indexes[torch.randperm(len(cat_indexes))[:cat_freq-lowest_freq]]
          indexes_to_delete = torch.cat((indexes_to_delete, new_indexes_to_delete))

    Idxs_to_keep = torch.tensor([i for i in range(len(Cat_labels)) if i not in indexes_to_delete], device = DEVICE) #the elements i will keep are the ones not present in indexes_to_delete
    # use torch.index_select() to select the elements to keep
    new_Cat_labels = torch.index_select(Cat_labels, 0, Idxs_to_keep)
    #Here below i will re-label the Cat_labels with more manageable names (i.e. progressives from 0 to nr categories -1, as happens for MNIST)
    proxy_cat = 2 #i begin from 2, given that labels 0 and 1 already exist in the old labelling system
    for category in cats:
       if category>=10: #i will change the name to all categories that are not 0 and 1
          new_Cat_labels = torch.where(new_Cat_labels == category, proxy_cat, new_Cat_labels) #this updates new_Cat_labels by replacing certain category labels (category) with a different value (proxy_cat) based on the specified condition (new_Cat_labels == category)
          proxy_cat = proxy_cat + 1

    new_Train_data = torch.index_select(Train_data, 0, Idxs_to_keep) #i select just the training examples corresponding to Idxs_to_keep
  else:
    new_Train_data = Train_data
    new_Cat_labels = Train_labels

  if for_classifier: # if you need your data (CelebA) to be preprocessed to be inputted to a classifier
    #then i apply the following transformation to make the data suitable to ResNet
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    dataset = MyDataset(features = new_Train_data, labels= new_Cat_labels, transform=transform)
    #dataset = MyDataset(Train_labels, Train_data)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    return train_loader #i return the loader ready to be used for classifier training or testing
  elif Old_rbm==False: 
    #This should be the classic preprocessing for the current code DBN
    new_Train_data = torch.squeeze(new_Train_data,1)
    new_Train_data = new_Train_data.view(new_Train_data.shape[0],new_Train_data.shape[1]*new_Train_data.shape[2])
    num_batches = new_Train_data.__len__() // Batch_size
    new_Train_data = new_Train_data[:(num_batches*Batch_size),:]
    new_Train_data = new_Train_data.view(num_batches,Batch_size,new_Train_data.shape[1])
    new_Cat_labels = new_Cat_labels[:(num_batches*Batch_size)]
    new_Cat_labels = new_Cat_labels.view(num_batches,Batch_size,1)
    train_dataset = {'data': new_Train_data, 'labels': new_Cat_labels}
    return train_dataset
  else: #if you are processing the data to be used by the old monolayer RBM (i.e. the one used in BI23), then...
    new_Train_data = new_Train_data.squeeze(1)
    new_Cat_labels = new_Cat_labels
    return new_Train_data, new_Cat_labels



def tool_loader_ZAMBRA(DEVICE,  selected_idx = [], half_data=False, only_data = True):
  from google.colab import drive
  drive.mount('/content/gdrive')
  Zambra_folder_drive = '/content/gdrive/My Drive/ZAMBRA_DBN/'

  #load the various files necessary
  with open(os.path.join(Zambra_folder_drive, 'cparams.json'), 'r') as filestream:
    CPARAMS = json.load(filestream)
  filestream.close()
  #and extract the relevant parameters
  DATASET_ID = CPARAMS['DATASET_ID']
  ALG_NAME = CPARAMS['ALG_NAME']

  READOUT        = CPARAMS['READOUT']
  RUNS           = CPARAMS['RUNS']
  LAYERS         = CPARAMS['LAYERS']
  NUM_DISCR      = CPARAMS['NUM_DISCR']
  INIT_SCHEME    = CPARAMS['INIT_SCHEME']

  with open(os.path.join(Zambra_folder_drive, f'lparams-{DATASET_ID.lower()}.json'), 'r') as filestream:
    LPARAMS = json.load(filestream)
  filestream.close()
  
  EPOCHS         = LPARAMS['EPOCHS']
  
  train_dataset_original, test_dataset_original = load_data_ZAMBRA(CPARAMS,LPARAMS,Zambra_folder_drive) #load the dataset of interest
  BATCH_SIZE = train_dataset_original['data'].shape[1]
  print('BATCH_SIZE '+ str(BATCH_SIZE))
  if 'CelebA' in DATASET_ID:
    if selected_idx == []:
        nrEx = train_dataset_original['labels'].shape[0] #usare
        train_dataset = copy.deepcopy(train_dataset_original)
        train_dataset['data'] = train_dataset['data'][:nrEx//2,:,:]  #for issues of processing on Colab, i will train using only half of the CelebA training data
        '''
        #Selection of a single label (UNUSED)
        cat_id = 20 #male
        L_all = deepcopy(train_dataset['labels'][:nrEx//2,:,:])
        train_dataset['labels'] = train_dataset['labels'][:nrEx//2,:,cat_id]
        test_dataset['labels'] = test_dataset['labels'][:,:,cat_id]
        '''
        train_dataset['labels'] = train_dataset['labels'][:nrEx//2,:,:]  # i downsample also the labels
        test_dataset = copy.deepcopy(test_dataset_original)
    else: 
        #i preprocess the data considering only the selected idxs
        train_dataset = Multiclass_dataset(train_dataset_original, selected_idx= selected_idx)
        print('train_dataset shape: '+ str(train_dataset['labels'].shape))
        test_dataset = Multiclass_dataset(test_dataset_original, selected_idx = selected_idx)
        print('test_dataset shape: '+ str(test_dataset['labels'].shape))
  else:
    train_dataset = train_dataset_original
    test_dataset = test_dataset_original
     
  if only_data: #if only the processed data are needed...
     return train_dataset_original, test_dataset_original


  if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor') #PyTorch will use GPU (CUDA) tensors as the default tensor type.

  Load_DBN_yn = int(input('Do you want to load a iDBN (Zambra 22 style) or do you want to train it? (1=yes, 0=no)'))
  
  if Load_DBN_yn == 0: #if the user chose to train a iDBN from scratch...
    #i divide the labels (Y) from the training examples (X), both for train and test
    Xtrain = train_dataset['data'].to(DEVICE)
    Xtest  = test_dataset['data'].to(DEVICE)
    Ytrain = train_dataset['labels'].to(DEVICE)
    Ytest  = test_dataset['labels'].to(DEVICE)


    # -----------------------------------------------------
    # Initialize performance metrics data structures
    loss_metrics = np.zeros((RUNS, EPOCHS, LAYERS))
    acc_metrics  = np.zeros((RUNS, EPOCHS, LAYERS))
    test_repr    = np.zeros((RUNS))
    test_reco    = np.zeros((RUNS))
    test_deno    = np.zeros((RUNS))
    Weber_fracs  = list()
    psycurves    = list()

    PATH_MODEL = os.getcwd()

    # Train the DBN
    for run in range(RUNS):
        
        print(f'\n\n---Run {run}\n')
        
        if ALG_NAME == 'g':
            
            dbn = gDBN(ALG_NAME, DATASET_ID, INIT_SCHEME, PATH_MODEL, EPOCHS).to(DEVICE)
            dbn.train(Xtrain, Xtest, Ytrain, Ytest, LPARAMS, readout = READOUT)
        elif ALG_NAME == 'i':
            
            dbn = iDBN(ALG_NAME, DATASET_ID, INIT_SCHEME, PATH_MODEL, EPOCHS).to(DEVICE)
            dbn.train(Xtrain, Xtest, Ytrain, Ytest, LPARAMS, readout = READOUT, num_discr = NUM_DISCR)
        elif ALG_NAME == 'fs':
            
            dbn = fsDBN(ALG_NAME, DATASET_ID, INIT_SCHEME, PATH_MODEL, EPOCHS).to(DEVICE)
            dbn.train(Xtrain, Xtest, Ytrain, Ytest, LPARAMS)
        #end
        
        for layer_id, rbm in enumerate(dbn.rbm_layers):
            loss_metrics[run, :, layer_id] = rbm.loss_profile
            acc_metrics[run, :, layer_id] = rbm.acc_profile
        #end
        
        if NUM_DISCR:
            Weber_fracs.append(dbn.Weber_fracs) # list of pd.DataFrame
            psycurves.append(dbn.psycurves)     # list of dicts with ratios and percs
        #end
        
        test_repr[run] = dbn.test(Xtest, Ytest)[0]
        dbn.DEVICE = DEVICE

        name = dbn.get_name()

        
        
        if not('CelebA' in DATASET_ID): #if you are not dealing with CelebA...
          dbn.Num_classes = 10 #i.e. the 10 classes of MNIST
          compute_inverseW_for_lblBiasing_ZAMBRA(dbn,train_dataset) #i compute the inverse weight matrix that i will use for label biasing on the top layer of the DBN
        elif not(selected_idx == []): #if you use CelebA with one-hot labels (i.e. 4 labels usually)
          dbn.Num_classes = 2**len(selected_idx) #the number of classes is 2 to the power of the selected classes 
          #compute_inverseW_for_lblBiasing_ZAMBRA(dbn,train_dataset,L = train_dataset['labels'])
          compute_inverseW_for_lblBiasing_ZAMBRA(dbn,train_dataset)
        else: #when you consider the multilabel case... (NOT USED)
          dbn.Num_classes = 40
          compute_inverseW_for_lblBiasing_ZAMBRA(dbn,train_dataset, L = train_dataset['labels'])
        fname = name +'_'+str(dbn.Num_classes)+'classes_nEp'+str(EPOCHS)+'_nL'+str(len(dbn.rbm_layers))+'_lastL'+str(dbn.top_layer_size)+'_bsz'+str(BATCH_SIZE)
        dbn.fname = fname
        #i save the trained DBN
        torch.save(dbn.to(torch.device('cpu')),
                    open(os.path.join(Zambra_folder_drive, f'{fname}.pkl'), 'wb'))
    #end
  else: #if you want to load an existing DBN...
    if not('CelebA' in DATASET_ID):
      Num_classes = 10
      nL = 3 #i.e. nr of layers
      last_layer_sz = str(1000)
    elif not(selected_idx == []):
      Num_classes = 2**len(selected_idx)
      nL = 3
      last_layer_sz = input('dimensione dell ultimo layer?') #usually 1000 o 5250
    else:
      Num_classes = 40
    fname = 'dbn_iterative_normal_'+DATASET_ID+'_'+str(Num_classes)+'classes_nEp'+str(EPOCHS)+'_nL'+str(nL)+'_lastL'+last_layer_sz+'_bsz'+str(BATCH_SIZE)
    dbn = torch.load(os.path.join(Zambra_folder_drive, fname+'.pkl'))
    if not(hasattr(dbn, 'fname')):
       dbn.fname = fname
       torch.save(dbn.to(torch.device('cpu')), open(os.path.join(Zambra_folder_drive, f'{fname}.pkl'), 'wb'))
  #load also the appropriate classifier to identify the samples generated by the DBN
  classifier = classifier_loader(dbn,train_dataset_original, test_dataset_original, selected_idx = selected_idx, DEVICE = 'cuda')
  return dbn,train_dataset_original, test_dataset_original,classifier

def classifier_loader(dbn,train_dataset_original, test_dataset_original, selected_idx = [], DEVICE = 'cuda'):
   if dbn.dataset_id == 'MNIST':
      #I create an instance of the classifier in which I will later load the saved parameters.
      classifier = VGG16((1,32,32), batch_norm=True).to(DEVICE) 
      PATH = '/content/gdrive/My Drive/VGG16_MNIST/VGG16_MNIST_best_val.pth'
      classifier.load_state_dict(torch.load(PATH))
   else: #CelebA
      Load_classifier = int(input('do you want to load a classifier or train it from scratch? (1=load, 0=train)'))
      num_classes = 2**len(selected_idx)
      fname = 'resnet_'+str(num_classes)+'classes.pt'

      if Load_classifier ==0: #if i want to train the classifier from scratch...
          #i adapt the dataloaders to be suitable for the classifier
          train_dataloader = Multiclass_dataset(train_dataset_original, selected_idx = selected_idx, for_classifier = True, Old_rbm=False, DEVICE ='cuda')
          test_dataloader = Multiclass_dataset(test_dataset_original, selected_idx = selected_idx, for_classifier = True, Old_rbm=False, DEVICE ='cuda')
          #i create the insance of the classifier and train it (all inside the CelebA_ResNet_classifier)
          classifier = CelebA_ResNet_classifier(ds_loaders = [train_dataloader, test_dataloader], num_classes = num_classes,  num_epochs = 20, learning_rate = 0.001, filename=fname)
      else: #if i want to load the classifier 
          classifier = CelebA_ResNet_classifier(ds_loaders = [],  num_classes = num_classes, filename=fname)

   classifier.eval() #i put the classifier in evaluation mode
   return classifier

def compute_inverseW_for_lblBiasing_ZAMBRA(model,train_dataset, L=[]):

    lbls = train_dataset['labels'].view(-1) # Flatten the labels in the training dataset
    Num_classes= model.Num_classes  # Get the number of classes from the model
    # Get the number of batches and batch size from the training dataset
    nr_batches = train_dataset['data'].shape[0]
    BATCH_SIZE = train_dataset['data'].shape[1]

    # If L is not provided, create a one-hot encoding matrix L for each label (i.e. num classes x examples)
    if L==[]:
      L = torch.zeros(Num_classes,lbls.shape[0], device = model.DEVICE)
      c=0
      for lbl in lbls:
          L[int(lbl),c]=1 #put =1 only the idx corresponding to the label of that example
          c=c+1
    else:
       L = L.view(40, -1) #for CelebA with all 40 labels (UNUSED)

    p_v, v = model(train_dataset['data'].cuda(), only_forward = True) #one step hidden layer of the training data by the model
    V_lin = v.view(nr_batches*BATCH_SIZE, model.top_layer_size)
    #I compute the inverse of the weight matrix of the linear classifier. weights_inv has shape (model.Num_classes x Hidden layer size (10 x 1000))
    weights_inv = torch.transpose(torch.matmul(torch.transpose(V_lin,0,1), torch.linalg.pinv(L)), 0, 1)

    model.weights_inv = weights_inv

    return weights_inv

def label_biasing_ZAMBRA(model, on_digits=1, topk = 149):

    # aim of this function is to implement the label biasing procedure described in
    # https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00515/full
    
    Num_classes=model.Num_classes
    # Now i set the label vector from which i will obtain the hidden layer of interest 
    Biasing_vec = torch.zeros (Num_classes,1, device = model.DEVICE)
    Biasing_vec[on_digits] = 1

    #I compute the biased hidden vector as the matmul of the trasposed weights_inv and the biasing vec. gen_hidden will have size (Hidden layer size x 1)
    gen_hidden= torch.matmul(torch.transpose(model.weights_inv,0,1), Biasing_vec)

    if topk>-1: #ATTENZIONE: label biasing con più di una label attiva (e.g. on_digits=[4,6]) funziona UNICAMENTE con topk>-1 (i.e. attivando le top k unità piu attive e silenziando le altre)
    #In caso contrario da errore CUDA non meglio specificato
      H = torch.zeros_like(gen_hidden, device = model.DEVICE) #crate an empty array of the same shape of gen_hidden 
      for c in range(gen_hidden.shape[1]): # for each example in gen_hidden...
        top_indices = torch.topk(gen_hidden[:,c], k=topk).indices # compute the most active indexes
        H[top_indices,c] = 1 #set the most active indexes to 0
      gen_hidden = H # gen_hidden is now binary (1 or 0)

    return gen_hidden


def generate_from_hidden_ZAMBRA(dbn, input_hid_prob, nr_gen_steps=1):
    #input_hid_prob has size Nr_hidden_units x num_cases. Therefore i transpose it
    input_hid_prob = torch.transpose(input_hid_prob,0,1)

    numcases = input_hid_prob.size()[0] #numbers of samples to generate
    hidden_layer_size = input_hid_prob.size()[1]
    vis_layerSize = dbn.rbm_layers[0].Nin
    # Initialize tensors to store hidden and visible probabilities and states
    # hid prob/states : nr layers x numbers of samples to generate x size of the hidden layer x number of generation steps
    # vis prob/states : numbers of samples to generate x size of the visible layer x number of generation steps
    hid_prob = torch.zeros(len(dbn.rbm_layers),numcases,hidden_layer_size, nr_gen_steps, device=dbn.DEVICE)
    hid_states = torch.zeros(len(dbn.rbm_layers), numcases,hidden_layer_size, nr_gen_steps, device=dbn.DEVICE)
    vis_prob = torch.zeros(numcases, vis_layerSize, nr_gen_steps, device=dbn.DEVICE)
    vis_states = torch.zeros(numcases ,vis_layerSize, nr_gen_steps, device=dbn.DEVICE)


    for gen_step in range(0, nr_gen_steps): #for each generation step...
      if gen_step==0: #if it is the 1st step of generation...
        hid_prob[2,:,:,gen_step]  = input_hid_prob #the hidden probability is the one in the input
        hid_states[2,:,:,gen_step]  = input_hid_prob
        c=1 # counter of layer depth
        for rbm in reversed(dbn.rbm_layers): #The reversed() function is used to reverse the order of elements in an iterable (e.g., a list )
            if c==1: #if it is the upper layer...
              p_v, v = rbm.backward(input_hid_prob) #compute the activity of the layer below using the biasing vector
              layer_size = v.shape[1]
              #i store the hid prob and state of the layer below
              hid_prob[2-c,:,:layer_size,gen_step]  = p_v
              hid_states[2-c,:,:layer_size,gen_step]  = v

            else:#if the layer selected is below the upper layer
              if c<len(dbn.rbm_layers): #if the layer selected is not the one above the visible layer (i.e. below there is another hidden layer)
                p_v, v = rbm.backward(v)
                layer_size = v.shape[1]
                #i store the hid prob and state of the layer below
                hid_prob[2-c,:,:layer_size,gen_step]  = p_v 
                hid_states[2-c,:,:layer_size,gen_step]  = v
              else: #if the layer below is the visible later
                v, p_v = rbm.backward(v) #passo la probabilità (che in questo caso è v) dopo
                layer_size = v.shape[1]
                #i store the visible state and probabilities
                vis_prob[:,:,gen_step]  = v 
                vis_states[:,:,gen_step]  = v
            c=c+1#for each layer i iterate, i update the counter
      else: #after the 1st gen step
            #from the visible state obtained in the previous activation, compute the activation of the upper layer
            for rbm in dbn.rbm_layers:
              p_v, v = rbm(v)
            #i store the probability and state of the upper layer
            hid_prob[2,:,:,gen_step]  = p_v 
            hid_states[2,:,:,gen_step]  = v
            #and i do the same as in the first step(code below)
            c=1
            for rbm in reversed(dbn.rbm_layers):

              if c<len(dbn.rbm_layers):
                p_v, v = rbm.backward(v)
                layer_size = v.shape[1]
                hid_prob[2-c,:,:layer_size,gen_step]  = p_v #the hidden probability is the one in the input
                hid_states[2-c,:,:layer_size,gen_step]  = v
              else:
                v, p_v = rbm.backward(v)
                layer_size = v.shape[1]
                vis_prob[:,:,gen_step]  = v #the hidden probability is the one in the input
                vis_states[:,:,gen_step]  = v
              c=c+1
              
    #the result dict will contain the output of the whole generation process
    result_dict = dict(); 
    result_dict['hid_states'] = hid_states
    result_dict['vis_states'] = vis_states
    result_dict['hid_prob'] = hid_prob
    result_dict['vis_prob'] = vis_prob

    return result_dict


class Intersection_analysis_ZAMBRA:
    def __init__(self, model, top_k_Hidden=100, nr_steps=100):
        self.model = model #the DBN model
        self.top_k_Hidden = top_k_Hidden #nr of hidden units with highest activity, which will then be binarized to 1
        self.nr_steps = nr_steps #nr steps of generation
        
    def do_intersection_analysis(self):
      #for the intersection method
      for dig in range(self.model.Num_classes): #for each class...
        g_H = label_biasing_ZAMBRA(self.model, on_digits=dig, topk = -1) #...do label biasing activating just that digit
        if dig == 0:
            hid_bias = g_H
        else:
            hid_bias = torch.hstack((hid_bias,g_H)) #stack together the label biasing vector for each digit

      vettore_indici_allDigits_biasing = torch.empty((0),device= self.model.DEVICE)

      for digit in range(self.model.Num_classes): #for each digit
        hid_vec_B = hid_bias[:,digit] #get the hidden state obtained by label biasing with the specific class 'digit'
        #in the next two lines i find the top p indexes in terms of activation
        top_values_biasing, top_idxs_biasing = torch.topk(hid_vec_B, self.top_k_Hidden) 
        vettore_indici_allDigits_biasing = torch.cat((vettore_indici_allDigits_biasing,top_idxs_biasing),0) #I concatenate the top p indexes for all digits in this vector

      unique_idxs_biasing,count_unique_idxs_biasing = torch.unique(vettore_indici_allDigits_biasing,return_counts=True) # Of the indexes found i take just the ones that are not repeated      

      digit_digit_common_elements_count_biasing = torch.zeros((self.model.Num_classes,self.model.Num_classes)) #in here i will count the number of common elements in each intersection
      self.unique_H_idxs_biasing = unique_idxs_biasing

      result_dict_biasing ={} #here i will store, for each combination of classes (keys), the units in intersection between them
      #for each category i iterate to compute the entries of the nr.classes x nr.classes matrices
      #itero per ogni digit per calcolare le entrate delle matrici 10 x 10
      for row in range(self.model.Num_classes): 
        for col in range(self.model.Num_classes):

          common_el_idxs_biasing = torch.empty((0),device= self.model.DEVICE)

          counter_biasing = 0
          for id in unique_idxs_biasing: #for each of the top indices
            digits_found = torch.floor(torch.nonzero(vettore_indici_allDigits_biasing==id)/self.top_k_Hidden)
            #torch.nonzero(vettore_indici_allDigits_biasing==id) finds the positions in the array vettore_indici_allDigits_biasing  where there is the value id is present
            #indeed, given that the vector vettore_indici_allDigits_biasing contains the top 100 most active units for each digit, if i divide the indexes by 100 (i.e. top_k_Hidden)
            #then i will find for which digit the unit id was active.

            if torch.any(digits_found==row) and torch.any(digits_found==col): #if the digits found present both the row and the col digits...
                common_el_idxs_biasing = torch.hstack((common_el_idxs_biasing,id)) #add the id to the vector of ids that will be used for intersection method biasing
                counter_biasing += 1 # i count the number of intersection elements to fill in the digit_digit_common_elements_count_biasing matrix

          result_dict_biasing[str(row)+','+str(col)] = common_el_idxs_biasing #store the units in the intersection
          digit_digit_common_elements_count_biasing[row,col] = counter_biasing

      self.result_dict_biasing = result_dict_biasing 

      print(digit_digit_common_elements_count_biasing)
      #lbl_bias_freqV = digit_digit_common_elements_count_biasing.view(100)/torch.sum(digit_digit_common_elements_count_biasing.view(100))

      return digit_digit_common_elements_count_biasing
    
    def generate_chimera_lbl_biasing(self,VGG_cl, elements_of_interest = [8,2], temperature=1, nr_of_examples = 1000, plot=0, entropy_correction=0):
      #this function does generation from chimeras obtained with the intersection method
      b_vec =torch.zeros(nr_of_examples,self.model.top_layer_size) 
      if not(elements_of_interest =='rand'): #if you don't want to generate from random chimeras
        dictionary_key = str(elements_of_interest[0])+','+str(elements_of_interest[1]) #entry of interest in the intersection dictionary
        b_vec[:,self.result_dict_biasing[dictionary_key].long()]=1#activate the entries corresponding the intersection units of interest

      else: #write 'rand' in elements of interest
        for i in range(nr_of_examples): #for every sample you want to generate
          #select two random classes
          n1 = random.randint(0, self.model.Num_classes-1) 
          n2 = random.randint(0, self.model.Num_classes-1)
          #activate the entries corresponding the intersection units of interest
          dictionary_key = str(n1)+','+str(n2) 
          b_vec[i,self.result_dict_biasing[dictionary_key].long()]=1

      b_vec = torch.transpose(b_vec,0,1)
      #b_vec = torch.unsqueeze(b_vec,0) #NOT USED
      d = generate_from_hidden_ZAMBRA(self.model, b_vec, nr_gen_steps=self.nr_steps) #generate from the hidden vectors produced
      
      d = Classifier_accuracy(d, VGG_cl, self.model, plot=plot, Thresholding_entropy=entropy_correction) #compute the accuracy of the classifier over the generation period
      df_average,df_sem, Transition_matrix_rowNorm = classification_metrics(d,self.model, Plot=plot, Ian=1)
      
      if nr_of_examples < 16:
        Plot_example_generated(d, self.model ,row_step = 10, dS=20, custom_steps = True, Show_classification = False)

      
      return d, df_average,df_sem, Transition_matrix_rowNorm
    
def Chimeras_nr_visited_states_ZAMBRA(model, VGG_cl, Ian =[], topk=149, apprx=1,plot=1,compute_new=1, nr_sample_generated =100, entropy_correction=[],cl_labels=[], lS=20):
    def save_mat_xlsx(my_array, filename='my_res.xlsx'):
        # create a pandas dataframe from the numpy array
        my_dataframe = pd.DataFrame(my_array)

        # save the dataframe as an excel file
        my_dataframe.to_excel(filename, index=False)
        # download the file
        files.download(filename)

    n_digits = model.Num_classes
    if Ian!=[]:
      fN='Visited_digits_k' + str(Ian.top_k_Hidden)+'.xlsx'
      fNerr='Visited_digits_error_k' + str(Ian.top_k_Hidden)+'.xlsx'
      fN_NDST='Nondigit_stateTime_k' + str(Ian.top_k_Hidden)+'.xlsx'
      fNerr_NDST='Nondigit_stateTime_error_k' + str(Ian.top_k_Hidden)+'.xlsx'
    else:
      fN='Visited_digits_Lbiasing_k' + str(topk)+'.xlsx'
      fNerr='Visited_digits_Lbiasing_error_k' + str(topk)+'.xlsx'
      fN_NDST='Nondigit_stateTime_Lbiasing_k' + str(topk)+'.xlsx'
      fNerr_NDST='Nondigit_stateTime_Lbiasing_error_k' + str(topk)+'.xlsx'

    if compute_new==1:
      #both
      Vis_states_mat = np.zeros((n_digits, n_digits))
      Vis_states_err = np.zeros((n_digits, n_digits))
      if n_digits==10:
        Non_digit_mat  = np.zeros((n_digits, n_digits))
        Non_digit_err  = np.zeros((n_digits, n_digits))

      if Ian!=[]:
        for row in range(n_digits):
          for col in range(row,n_digits):
            d, df_average,df_sem, Transition_matrix_rowNorm = Ian.generate_chimera_lbl_biasing(VGG_cl,elements_of_interest = [row,col], nr_of_examples = nr_sample_generated, temperature = 1, plot=0, entropy_correction= entropy_correction)
            Vis_states_mat[row,col]=df_average.Nr_visited_states[0]
            Vis_states_err[row,col]=df_sem.Nr_visited_states[0]
            if n_digits==10:
              Non_digit_mat[row,col] = df_average['Non-digit'][0]
              Non_digit_err[row,col] = df_sem['Non-digit'][0]
      else:
        numbers = list(range(n_digits))
        combinations_of_two = list(combinations(numbers, 2))

        for idx, combination in enumerate(combinations_of_two):
          gen_hidden = label_biasing_ZAMBRA(model, on_digits=  list(combination), topk = topk)
          gen_hidden_rep = gen_hidden.repeat(1,nr_sample_generated)
          d = generate_from_hidden_ZAMBRA(model, gen_hidden_rep , nr_gen_steps=100)
          d = Classifier_accuracy(d, VGG_cl,model, Thresholding_entropy=entropy_correction, labels=[], Batch_sz= 100, plot=0, dS=30, l_sz=3)
          df_average,df_sem, Transition_matrix_rowNorm = classification_metrics(d,model,Plot=0,dS=50,Ian=1)
          Vis_states_mat[combination[0],combination[1]]=df_average.Nr_visited_states[0]
          Vis_states_err[combination[0],combination[1]]=df_sem.Nr_visited_states[0]
          if n_digits==10:
            Non_digit_mat[combination[0],combination[1]] = df_average['Non-digit'][0]
            Non_digit_err[combination[0],combination[1]] = df_sem['Non-digit'][0]


      save_mat_xlsx(Vis_states_mat, filename=fN)
      save_mat_xlsx(Vis_states_err, filename=fNerr)
      if n_digits==10:
        save_mat_xlsx(Non_digit_mat, filename=fN_NDST)
        save_mat_xlsx(Non_digit_err, filename=fNerr_NDST)

    else: #load already computed Vis_states_mat
      if n_digits==10:
        Non_digit_mat = pd.read_excel(fN_NDST)
        Non_digit_err = pd.read_excel(fNerr_NDST)
        # Convert the DataFrame to a NumPy array
        Non_digit_mat = Non_digit_mat.values
        Non_digit_err = Non_digit_err.values
      Vis_states_mat = pd.read_excel(fN)
      # Convert the DataFrame to a NumPy array
      Vis_states_mat = Vis_states_mat.values

      Vis_states_err = pd.read_excel(fNerr)
      # Convert the DataFrame to a NumPy array
      Vis_states_err = Vis_states_err.values

    if plot==1:

      Vis_states_mat = Vis_states_mat.round(apprx)
      Vis_states_err = Vis_states_err.round(apprx)

      plt.figure(figsize=(15, 15))
      mask = np.triu(np.ones_like(Vis_states_mat),k=+1) # k=+1 per rimuovere la diagonale
      # Set the lower triangle to NaN
      Vis_states_mat = np.where(mask==0, np.nan, Vis_states_mat)
      Vis_states_mat = Vis_states_mat.T
      #ax = sns.heatmap(Vis_states_mat, linewidth=0.5, annot=False,square=True, cbar=False)
      ax = sns.heatmap(Vis_states_mat, linewidth=0.5, annot=True, annot_kws={"size": lS},square=True,cbar_kws={"shrink": .82}, fmt='.1f', cmap='jet')
      if not(cl_labels==[]):
        ax.set_xticklabels(cl_labels)
        ax.set_yticklabels(cl_labels)
      #ax.set_xticklabels(T_mat_labels)
      ax.tick_params(axis='both', labelsize=lS)

      plt.xlabel('Class', fontsize = lS) # x-axis label with fontsize 15
      plt.ylabel('Class', fontsize = lS) # y-axis label with fontsize 15
      #cbar = plt.gcf().colorbar(ax.collections[0], location='left', shrink=0.82)
      cbar = ax.collections[0].colorbar
      cbar.ax.tick_params(labelsize=lS)
      plt.show()

    if n_digits==10:
      return Vis_states_mat, Vis_states_err,Non_digit_mat,Non_digit_err
    else:
      return Vis_states_mat, Vis_states_err


def Perc_H_act(model, sample_labels, gen_data_dictionary=[], dS = 50, l_sz = 5, layer_of_interest=2):

    c=0 #inizializzo il counter per cambiamento colore
    cmap = cm.get_cmap('hsv') # inizializzo la colormap che utilizzerò per il plotting
    figure, axis = plt.subplots(1, 1, figsize=(15,15)) #setto le dimensioni della figura
    lbls = [] # qui storo le labels x legenda

    for digit in range(model.Num_classes): # per ogni digit...
        
        Color = cmap(c/256) #setto il colore di quel determinato digit
        l = torch.where(sample_labels == digit) #trovo gli indici dei test data che contengono quel determinato digit
        nr_examples= len(l[0]) #nr degli esempi di quel digit (i.e. n)

        gen_H_digit = gen_data_dictionary['hid_states'][layer_of_interest,l[0],:,:]
        nr_steps = gen_H_digit.size()[2]
        if digit == 0:
            Mean_storing = torch.zeros(model.Num_classes,nr_steps, device = 'cuda')
            Sem_storing = torch.zeros(model.Num_classes,nr_steps, device = 'cuda')
        SEM = torch.std(torch.mean(gen_H_digit,1)*100,0)/math.sqrt(nr_examples)
        MEAN = torch.mean(torch.mean(gen_H_digit,1)*100,0).cpu()
        Mean_storing[digit, : ] = MEAN.cuda()
        Sem_storing[digit, : ] = SEM

        if digit==0: #evito di fare sta operazione più volte
          y_lbl = '% active H units'

        SEM = SEM.cpu() #sposto la SEM su CPU x plotting
        x = range(1,nr_steps+1) #asse delle x, rappresentante il nr di step di ricostruzione svolti
        plt.plot(x, MEAN, c = Color, linewidth=l_sz) #plotto la media
        plt.fill_between(x,MEAN-SEM, MEAN+SEM, color=Color, alpha=0.3) # e le barre di errore
        
        c = c+25
        lbls.append(digit)

    axis.legend(lbls, bbox_to_anchor=(1.04,1), loc="upper left", fontsize=dS) # legenda
    #ridimensiono etichette assi e setto le labels
    axis.tick_params(axis='x', labelsize= dS) 
    axis.tick_params(axis='y', labelsize= dS)
    axis.set_ylabel(y_lbl,fontsize=dS)
    axis.set_xlabel('Generation step',fontsize=dS)
    axis.set_title(y_lbl+' - digitwise',fontsize=dS)


    axis.set_ylim([0,100])
    return Mean_storing, Sem_storing

def Plot_example_generated(input_dict,num_classes = 10,row_step = 10, dS=50, lblpad = 110, custom_steps = True, Show_classification = False, not_random_idxs = True):
    
    Generated_samples=input_dict['vis_states']
    nr_steps = Generated_samples.shape[2]

    img_side = int(np.sqrt(Generated_samples.shape[1]))


    if Show_classification ==True:
      Classifications = input_dict['Cl_pred_matrix']
    
    if custom_steps == True:
      steps=[3,5,10,25,50,100]
      rows=len(steps)
    else:
      steps = range(row_step,nr_steps+1,row_step) #controlla che funzioni
      rows = math.floor(nr_steps/row_step) 

    cols = Generated_samples.shape[0]
    fig_side = 25/num_classes
    if cols>num_classes:
      figure, axis = plt.subplots(rows+1,num_classes, figsize=(25*(num_classes/num_classes),fig_side*(1+rows)))
    elif cols>1:
      figure, axis = plt.subplots(rows+1,cols, figsize=(25*(cols/num_classes),fig_side*(1+rows)))
    else:
      figure, axis = plt.subplots(rows+1,cols+1, figsize=(25*(cols/num_classes),fig_side*(1+rows)))

    if cols >= 10:
      if not_random_idxs ==True:
        random_numbers = range(num_classes)
      else:
        random_numbers = random.sample(range(cols), num_classes) # 10 random samples are selected
    else:
      random_numbers = random.sample(range(cols), cols) # 10 random samples are selected


    c=0
    for sample_idx in random_numbers: #per ogni sample selezionato

        # plotto la ricostruzione dopo uno step

        reconstructed_img= Generated_samples[sample_idx,:,0] #estraggo la prima immagine ricostruita per il particolare esempio (lbl può essere un nome un po fuorviante)
        reconstructed_img = reconstructed_img.view((img_side,img_side)).cpu() #ridimensiono l'immagine e muovo su CPU
        axis[0, c].tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
        axis[0, c].imshow(reconstructed_img , cmap = 'gray')
        if Show_classification==True:
          axis[0, c].set_title("Class {}".format(Classifications[sample_idx,0]), fontsize=dS)
        if c==0:
          ylabel = axis[0, c].set_ylabel("Step {}".format(1), fontsize=dS,rotation=0, labelpad=lblpad)

        axis[0, c].set_xticklabels([])
        axis[0, c].set_yticklabels([])
        axis[0, c].set_aspect('equal')

        #for idx,step in enumerate(range(row_step,nr_steps+1,row_step)): # idx = riga dove plotterò, step è il recostruction step che ci plotto
        for idx,step in enumerate(steps): # idx = riga dove plotterò, step è il recostruction step che ci plotto
            idx = idx+1 #sempre +1 perchè c'è sempre 1 step reconstruction

            #plotto la ricostruzione

            reconstructed_img= Generated_samples[sample_idx,:,step-1] #step-1 perchè 0 è la prima ricostruzione
            reconstructed_img = reconstructed_img.view((img_side,img_side)).cpu()
            axis[idx, c].tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
            axis[idx, c].imshow(reconstructed_img , cmap = 'gray')
            if Show_classification==True:
              axis[idx, c].set_title("Class {}".format(Classifications[sample_idx,step-1]), fontsize=dS)
            #axis[idx, lbl].set_title("Step {}".format(step) , fontsize=dS)
            if c==0:
              ylabel = axis[idx, c].set_ylabel("Step {}".format(step), fontsize=dS, rotation=0, labelpad=lblpad)
              


            axis[idx, c].set_xticklabels([])
            axis[idx, c].set_yticklabels([])
            axis[idx, c].set_aspect('equal')

        c=c+1

    #aggiusto gli spazi tra le immagini
    plt.subplots_adjust(left=0.1, 
                        bottom=0.1,  
                        right=0.9,  
                        top=0.9,  
                        wspace=0.4,  
                        hspace=0.2) 
    
    #plt.savefig("Reconstuct_plot.jpg") #il salvataggio è disabilitato

    plt.show()



def readout_V_to_Hlast(dbn,train_dataset,test_dataset, DEVICE='cuda'):
  if 'CelebA' in dbn.dataset_id:
    train_dataset = Multiclass_dataset(train_dataset, selected_idx= [20,31])
    test_dataset = Multiclass_dataset(test_dataset, selected_idx = [20,31])

  Xtrain = train_dataset['data'].to(DEVICE)
  Xtest  = test_dataset['data'].to(DEVICE)
  Ytrain = train_dataset['labels'].to(DEVICE)
  Ytest  = test_dataset['labels'].to(DEVICE)
  readout_acc_V =[]

  n_train_batches = Xtrain.shape[0]
  n_test_batches = Xtest.shape[0]
  batch_size = Xtrain.shape[1]

  readout_acc = dbn.rbm_layers[1].get_readout(Xtrain, Xtest, Ytrain, Ytest)
  print(f'Readout accuracy = {readout_acc*100:.2f}')
  readout_acc_V.append(readout_acc)
  for rbm in dbn.rbm_layers:

      _Xtrain = torch.zeros((n_train_batches, batch_size, rbm.Nout))
      _Xtest = torch.zeros((n_test_batches, batch_size, rbm.Nout))

      _Xtest, _ = rbm(Xtest)

      batch_indices = list(range(n_train_batches))
      random.shuffle(batch_indices)
      with tqdm(batch_indices, unit = 'Batch') as tlayer:
          for idx, n in enumerate(tlayer):

              tlayer.set_description(f'Layer {rbm.layer_id}')
              _Xtrain[n,:,:], _ = rbm(Xtrain[n,:,:])

          #end BATCHES
      #end WITH

      readout_acc = rbm.get_readout(_Xtrain, _Xtest, Ytrain, Ytest)
      print(f'Readout accuracy = {readout_acc*100:.2f}')
      #end
      readout_acc_V.append(readout_acc)

      Xtrain = _Xtrain.clone()
      Xtest  = _Xtest.clone()
  return readout_acc_V


def comparisons_plot(Results_dict, sel_key='Nr_visited_states_MEAN'):
  # parametri grafici
  def Nr_visited_states_MEAN_SEM(Results_dict, MNIST_fname):
    def SEM(measure):
      import math
      nr_of_measures = len(measure)
      if not(isinstance(measure, np.ndarray)):
        measure = np.asarray(measure)
      sem = np.std(measure)/math.sqrt(nr_of_measures)
      return sem

    MNIST = Results_dict[MNIST_fname]
    if not('Nr_visited_states_MEAN' in MNIST):
      if len(MNIST.keys()) == 3:
        sz_mean = len(MNIST.keys())
      else:
        sz_mean = len(MNIST.keys())-1
      Nr_visited_states_MEAN = np.zeros(sz_mean)
      Nr_visited_states_SEM = np.zeros(sz_mean)
      c=0
      keys = ['Nr_visited_states_LB', 'Nr_visited_states_C2lb', 'Nr_visited_states_Cint']
      for k in keys:
        print()
        MNIST[k] = np.array(MNIST[k])
        print(MNIST[k])
        Nr_visited_states_MEAN[c]=np.mean(MNIST[k])
        Nr_visited_states_SEM[c]=SEM(MNIST[k])
        c=c+1
      Results_dict[MNIST_fname]['Nr_visited_states_MEAN']=Nr_visited_states_MEAN
      Results_dict[MNIST_fname]['Nr_visited_states_SEM']=Nr_visited_states_SEM
    return Results_dict

  LineW = 4
  Mk = 's'
  Mk_sz = 12
  Cp_sz = 12
  Err_bar_sz = 4
  Scritte_sz = 50

  if sel_key=='Nr_visited_states_MEAN':
    x_labels = ['LB', 'C_2LB', 'C_int']
    x_lab = 'Generation method'
    y_lab = 'Number of states'
    y_r = [1,5]
  else:
    # Create a list of the x-axis labels
    x_labels = ['V', 'H1', 'H2', 'H3']
    x_lab = 'Layer'
    y_lab = 'Accuracy'
    y_r = [0.85,1]


  # ottieni le chiavi del dizionario e calcola il loro numero
  keys = list(Results_dict.keys())
  n = len(keys)

  # costruisci la stringa per il prompt di input
  input_prompt = "Che chiavi vuoi selezionare?\n"
  for i in range(n):
      input_prompt += f"{i}: {keys[i]}\n"

  # richiedi all'utente di selezionare una chiave
  selected_key_index = input(input_prompt)
  selected_key_index = eval(selected_key_index)

  fnames = [keys[idx] for idx in selected_key_index]
  def custom_sort(elem):
    if 'dbn' in elem:
        return (0, -1 * elem.count('MNIST'))
    elif 'MNIST' in elem:
        return (1, -1 * elem.count('MNIST'))
    else:
        return (2, 0)

  fnames = sorted(fnames, key=custom_sort)

  line_list = []
  fig, ax = plt.subplots(figsize=(15, 15))
  for fname in fnames:
    Results_dict = Nr_visited_states_MEAN_SEM(Results_dict, fname)
    Dati = np.array(Results_dict[fname][sel_key])
    if 'RBM' in fname:
      model_type = 'RBM'
      L_style='--'
    else:
      model_type = 'iDBN'
      L_style='-'
    if 'MNIST' in fname:
      L_col = 'blue'
      ds_type = 'MNIST'
    else:
      L_col = 'red'
      ds_type = 'CelebA'
      if sel_key=='readout':
        y_r = [0.75,0.8]
    print(Dati)
    model_type = model_type+' '+ds_type
    linei, = ax.plot(x_labels, Dati, color=L_col, label=model_type, linewidth=LineW, marker=Mk, markersize=Mk_sz, linestyle=L_style)
    line_list.append(linei)
    if sel_key=='Nr_visited_states_MEAN':
      Dati_SEM = np.array(Results_dict[fname]['Nr_visited_states_SEM'])
      # Add error bars to the second line
      ax.errorbar(x_labels, Dati, yerr=Dati_SEM, fmt='none', ecolor=L_col, capsize=Cp_sz,  elinewidth=Err_bar_sz)
  if len(fnames)>1:
    ax.legend(handles=line_list, loc='upper center', bbox_to_anchor=(1.35, 0.7), fontsize=Scritte_sz)


    
  # Set the x-axis label
  ax.set_xlabel(x_lab, fontsize=Scritte_sz)
  # Set the y-axis label
  ax.set_ylabel(y_lab, fontsize=Scritte_sz)
  # Set the font size of all the text in the plot
  plt.rc('font', size=Scritte_sz)
  # Set the y-axis range
  ax.set_ylim(y_r)
  # Set the x-axis tick font size
  ax.tick_params(axis='x', labelsize=Scritte_sz)
  # Set the y-axis tick font size
  ax.tick_params(axis='y', labelsize=Scritte_sz)
  # Display the plot
  plt.show()

