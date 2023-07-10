from tqdm import tqdm
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


def load_data_ZAMBRA(CPARAMS,LPARAMS,Zambra_folder_drive):
    # Riassume il caricamento dei dati del CelebA nella repository di Zambra, evitando problemi ed errori
    DATASET_ID = CPARAMS['DATASET_ID']
    n_cols_labels = 1
    if DATASET_ID =='MNIST':
       NUM_FEAT= np.int32(28*28)
    elif DATASET_ID =='CIFAR10':
      NUM_FEAT= np.int32(32*32)
    elif 'CelebA' in DATASET_ID:
       NUM_FEAT= np.int32(64*64)
       n_cols_labels = 40


    BATCH_SIZE = LPARAMS['BATCH_SIZE']
    test_filename = 'test_dataset_'+DATASET_ID+'.npz'
    train_filename = 'train_dataset_'+DATASET_ID+'.npz'
    trainfile_path= os.path.join(Zambra_folder_drive,'dataset_dicts',train_filename)
    testfile_path = os.path.join(Zambra_folder_drive,'dataset_dicts',test_filename)

    if os.path.exists(trainfile_path):
      train_dataset = dict(np.load(trainfile_path))
      test_dataset = dict(np.load(testfile_path))
      # Convert the numpy arrays to torch tensors
      for key in train_dataset:
          train_dataset[key] = torch.from_numpy(train_dataset[key])
          test_dataset[key]= torch.from_numpy(test_dataset[key])
      Real_BATCH_SIZE = train_dataset['data'].shape[1]
      if not(Real_BATCH_SIZE ==BATCH_SIZE):
        reshape_yn = int(input('data found with batchsize '+str(Real_BATCH_SIZE)+ '.Reshape it to the desired batchsize '+str(BATCH_SIZE)+'? (1=y,0=n)'))
        if reshape_yn==1:
          def reshape_data(train_dataset, Real_BATCH_SIZE, BATCH_SIZE):
              n = BATCH_SIZE//Real_BATCH_SIZE #pensato solo per 128 e 64
              if not(train_dataset['data'].shape[0]%n==0):
                  train_dataset['data'] = train_dataset['data'][:-1,:,:]
                  train_dataset['labels'] = train_dataset['labels'][:-1,:,:]
              train_dataset['data'] = train_dataset['data'].view(train_dataset['data'].shape[0]//n, BATCH_SIZE, train_dataset['data'].shape[2])
              train_dataset['labels'] = train_dataset['labels'].view(train_dataset['labels'].shape[0]//n, BATCH_SIZE, train_dataset['labels'].shape[2])
              return train_dataset
          train_dataset = reshape_data(train_dataset, Real_BATCH_SIZE, BATCH_SIZE)
          test_dataset = reshape_data(test_dataset, Real_BATCH_SIZE, BATCH_SIZE)
    else:
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

      def data_and_labels(data_train, BATCH_SIZE,NUM_FEAT):
        train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
        num_batches = data_train.__len__() // BATCH_SIZE
        train_data = torch.empty(num_batches, BATCH_SIZE, NUM_FEAT)
        train_labels = torch.empty(num_batches, BATCH_SIZE, n_cols_labels)
        with tqdm(train_loader, unit = 'Batch') as tdata:
            for idx, (batch, labels) in enumerate(tdata):
                tdata.set_description(f'Train Batch {idx}\t')
                if idx<num_batches:
                  bsize = batch.shape[0]
                  if DATASET_ID =='MNIST':
                    train_data[idx,:,:] = batch.reshape(bsize, -1).type(torch.float32)
                  else:
                    gray_batch = batch.mean(dim=1, keepdim=True) # converti in scala di grigi
                    for i in range(bsize):
                        img_to_store = gray_batch[i].numpy().squeeze()
                        if 'BW' in DATASET_ID:
                          threshold = threshold_sauvola(img_to_store,window_size=7, k=0.05)
                          img_to_store = img_to_store > threshold
                        train_data[idx, i, :] = torch.from_numpy(img_to_store.reshape(-1).astype(np.float32))
                  if len(labels.shape)==1:
                     labels = labels.unsqueeze(1)
                  train_labels[idx, :, :] = labels.type(torch.float32)
        return train_data, train_labels   
      
      train_data, train_labels = data_and_labels(data_train, BATCH_SIZE,NUM_FEAT)
      test_data, test_labels = data_and_labels(data_test, BATCH_SIZE,NUM_FEAT)

      train_dataset = {'data': train_data, 'labels': train_labels}
      test_dataset = {'data': test_data, 'labels': test_labels}
      Path(os.path.join(Zambra_folder_drive,'dataset_dicts')).mkdir(exist_ok=True)
      np.savez(trainfile_path, **train_dataset)
      np.savez(testfile_path, **test_dataset)

    return train_dataset, test_dataset

class MyDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        # apply transformations if specified
        if self.transform:
            feature = self.transform(feature)

        return feature, label

def Multiclass_dataset(train_dataset, selected_idx = [20,31], for_classifier = False, Old_rbm=False, DEVICE ='cuda'):
  Batch_size = train_dataset['data'].shape[1]
  Train_data = copy.deepcopy(train_dataset['data']).to(DEVICE)
  if not(selected_idx==[]):
    Train_labels = copy.deepcopy(train_dataset['labels'][:,:,selected_idx]).to(DEVICE)
  else:
    Train_labels = copy.deepcopy(train_dataset['labels']).to(DEVICE)
  side_img = int(np.sqrt(Train_data.shape[2]))
  Train_data = Train_data.view(Train_data.shape[0]*Train_data.shape[1], side_img, side_img).unsqueeze(1)
  Train_labels = Train_labels.view(Train_labels.shape[0]*Train_labels.shape[1], Train_labels.shape[2])


  if not(selected_idx==[]):
    powers_of_10 = torch.pow(10, torch.arange(len(selected_idx), dtype=torch.float)).to(DEVICE)
    Cat_labels = torch.matmul(Train_labels,powers_of_10)

    cats, f_cat = torch.unique(Cat_labels, return_counts= True)
    lowest_freq_idx = torch.argmin(f_cat)
    lowest_freq = f_cat[lowest_freq_idx]
    lowest_freq_cat = cats[lowest_freq_idx]

    for category in cats:
      if not(category==lowest_freq_cat):
        cat_freq = f_cat[cats==category]
        cat_indexes = torch.where(Cat_labels == category)[0]
        if category==0:
          indexes_to_delete = cat_indexes[torch.randperm(len(cat_indexes))[:cat_freq-lowest_freq]]
        else:
          new_indexes_to_delete = cat_indexes[torch.randperm(len(cat_indexes))[:cat_freq-lowest_freq]]
          indexes_to_delete = torch.cat((indexes_to_delete, new_indexes_to_delete))

    Idxs_to_keep = torch.tensor([i for i in range(len(Cat_labels)) if i not in indexes_to_delete], device = DEVICE)
    # use torch.index_select() to select the elements to keep
    new_Cat_labels = torch.index_select(Cat_labels, 0, Idxs_to_keep)
    proxy_cat = 2
    for category in cats:
       if category>=10:
          new_Cat_labels = torch.where(new_Cat_labels == category, proxy_cat, new_Cat_labels)
          proxy_cat = proxy_cat + 1

    new_Train_data = torch.index_select(Train_data, 0, Idxs_to_keep)
  else:
    new_Train_data = Train_data
    new_Cat_labels = Train_labels

  if for_classifier:
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    dataset = MyDataset(features = new_Train_data, labels= new_Cat_labels, transform=transform)
    #dataset = MyDataset(Train_labels, Train_data)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    return train_loader
  elif Old_rbm==False:
    new_Train_data = torch.squeeze(new_Train_data,1)
    new_Train_data = new_Train_data.view(new_Train_data.shape[0],new_Train_data.shape[1]*new_Train_data.shape[2])
    num_batches = new_Train_data.__len__() // Batch_size
    new_Train_data = new_Train_data[:(num_batches*Batch_size),:]
    new_Train_data = new_Train_data.view(num_batches,Batch_size,new_Train_data.shape[1])
    new_Cat_labels = new_Cat_labels[:(num_batches*Batch_size)]
    new_Cat_labels = new_Cat_labels.view(num_batches,Batch_size,1)
    train_dataset = {'data': new_Train_data, 'labels': new_Cat_labels}
    return train_dataset
  else:
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

  train_dataset_original, test_dataset_original = load_data_ZAMBRA(CPARAMS,LPARAMS,Zambra_folder_drive)
  BATCH_SIZE = train_dataset_original['data'].shape[1]
  print('BATCH_SIZE '+ str(BATCH_SIZE))
  if 'CelebA' in DATASET_ID:
    if selected_idx == []:
        nrEx = train_dataset_original['labels'].shape[0] #usare
        #cat_id = 20 #male
        train_dataset = copy.deepcopy(train_dataset_original)
        train_dataset['data'] = train_dataset['data'][:nrEx//2,:,:]  #usare
        #L_all = deepcopy(train_dataset['labels'][:nrEx//2,:,:])
        #train_dataset['labels'] = train_dataset['labels'][:nrEx//2,:,cat_id]
        #test_dataset['labels'] = test_dataset['labels'][:,:,cat_id]
        train_dataset['labels'] = train_dataset['labels'][:nrEx//2,:,:]  #usare
        test_dataset = copy.deepcopy(test_dataset_original)
    else: 
        train_dataset = Multiclass_dataset(train_dataset_original, selected_idx= selected_idx)
        print('train_dataset shape: '+ str(train_dataset['labels'].shape))
        test_dataset = Multiclass_dataset(test_dataset_original, selected_idx = selected_idx)
        print('test_dataset shape: '+ str(test_dataset['labels'].shape))
  else:
    train_dataset = train_dataset_original
    test_dataset = test_dataset_original
     

    #HALF DATA è Provvisorio
  if only_data:
     return train_dataset_original, test_dataset_original


  if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

  Load_DBN_yn = int(input('Do you want to load a iDBN (Zambra 22 style) or do you want to train it? (1=yes, 0=no)'))
  
  if Load_DBN_yn == 0:
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

        
        
        if not('CelebA' in DATASET_ID):
          dbn.Num_classes = 10
          compute_inverseW_for_lblBiasing_ZAMBRA(dbn,train_dataset)
        elif not(selected_idx == []):
          dbn.Num_classes = 2**len(selected_idx)
          #compute_inverseW_for_lblBiasing_ZAMBRA(dbn,train_dataset,L = train_dataset['labels'])
          compute_inverseW_for_lblBiasing_ZAMBRA(dbn,train_dataset)
        else:
          dbn.Num_classes = 40
          compute_inverseW_for_lblBiasing_ZAMBRA(dbn,train_dataset, L = train_dataset['labels'])
        fname = name +'_'+str(dbn.Num_classes)+'classes_nEp'+str(EPOCHS)+'_nL'+str(len(dbn.rbm_layers))+'_lastL'+str(dbn.top_layer_size)+'_bsz'+str(BATCH_SIZE)
        dbn.fname = fname
        torch.save(dbn.to(torch.device('cpu')),
                    open(os.path.join(Zambra_folder_drive, f'{fname}.pkl'), 'wb'))
    #end
  else:
    if not('CelebA' in DATASET_ID):
      Num_classes = 10
      nL = 3
      last_layer_sz = str(1000)
    elif not(selected_idx == []):
      Num_classes = 2**len(selected_idx)
      nL = 3
      last_layer_sz = input('dimensione dell ultimo layer?')
    else:
      Num_classes = 40
    fname = 'dbn_iterative_normal_'+DATASET_ID+'_'+str(Num_classes)+'classes_nEp'+str(EPOCHS)+'_nL'+str(nL)+'_lastL'+last_layer_sz+'_bsz'+str(BATCH_SIZE)
    dbn = torch.load(os.path.join(Zambra_folder_drive, fname+'.pkl'))
    if not(hasattr(dbn, 'fname')):
       dbn.fname = fname
       torch.save(dbn.to(torch.device('cpu')), open(os.path.join(Zambra_folder_drive, f'{fname}.pkl'), 'wb'))
       
  classifier = classifier_loader(dbn,train_dataset_original, test_dataset_original, selected_idx = selected_idx, DEVICE = 'cuda')
  return dbn,train_dataset_original, test_dataset_original,classifier

def classifier_loader(dbn,train_dataset_original, test_dataset_original, selected_idx = [], DEVICE = 'cuda'):
   if dbn.dataset_id == 'MNIST':
      classifier = VGG16((1,32,32), batch_norm=True).to(DEVICE) #creo un'istanza del classificatore in cui poi caricherò i parametri salvati
      PATH = '/content/gdrive/My Drive/VGG16_MNIST/VGG16_MNIST_best_val.pth'
      classifier.load_state_dict(torch.load(PATH))
   else:
      Load_classifier = int(input('do you want to load a classifier or train it from scratch? (1=load, 0=train)'))
      num_classes = 2**len(selected_idx)
      fname = 'resnet_'+str(num_classes)+'classes.pt'

      if Load_classifier ==0:
          train_dataloader = Multiclass_dataset(train_dataset_original, selected_idx = selected_idx, for_classifier = True, Old_rbm=False, DEVICE ='cuda')
          test_dataloader = Multiclass_dataset(test_dataset_original, selected_idx = selected_idx, for_classifier = True, Old_rbm=False, DEVICE ='cuda')
          classifier = CelebA_ResNet_classifier(ds_loaders = [train_dataloader, test_dataloader], num_classes = num_classes,  num_epochs = 20, learning_rate = 0.001, filename=fname)
      else:
          classifier = CelebA_ResNet_classifier(ds_loaders = [],  num_classes = num_classes, filename=fname)

   classifier.eval()
   return classifier

def compute_inverseW_for_lblBiasing_ZAMBRA(model,train_dataset, L=[]):

    lbls = train_dataset['labels'].view(-1)
    Num_classes= model.Num_classes
    nr_batches = train_dataset['data'].shape[0]
    BATCH_SIZE = train_dataset['data'].shape[1]
    if L==[]:
      L = torch.zeros(Num_classes,lbls.shape[0], device = model.DEVICE)
      c=0
      for lbl in lbls:
          L[int(lbl),c]=1
          c=c+1
    else:
       L = L.view(40, -1)

    p_v, v = model(train_dataset['data'].cuda(), only_forward = True)
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
          H = torch.zeros_like(gen_hidden, device = model.DEVICE) #crea un array vuoto della forma di gen_hidden
          for c in range(gen_hidden.shape[1]): # per ciascun esempio di gen_hidden...
            top_indices = torch.topk(gen_hidden[:,c], k=topk).indices # computa gli indici più attivati
            H[top_indices,c] = 1 #setta gli indici più attivi a 1
          gen_hidden = H # gen_hidden ha ora valori binari (1 e 0)



        return gen_hidden


def generate_from_hidden_ZAMBRA(dbn, input_hid_prob, nr_gen_steps=1):
    #input_hid_prob has size Nr_hidden_units x num_cases. Therefore i transpose it
    input_hid_prob = torch.transpose(input_hid_prob,0,1)

    numcases = input_hid_prob.size()[0] #numbers of samples to generate
    hidden_layer_size = input_hid_prob.size()[1]
    vis_layerSize = dbn.rbm_layers[0].Nin

    hid_prob = torch.zeros(len(dbn.rbm_layers),numcases,hidden_layer_size, nr_gen_steps, device=dbn.DEVICE)
    hid_states = torch.zeros(len(dbn.rbm_layers), numcases,hidden_layer_size, nr_gen_steps, device=dbn.DEVICE)
    vis_prob = torch.zeros(numcases, vis_layerSize, nr_gen_steps, device=dbn.DEVICE)
    vis_states = torch.zeros(numcases ,vis_layerSize, nr_gen_steps, device=dbn.DEVICE)


    for gen_step in range(0, nr_gen_steps):
      if gen_step==0: #if it is the 1st step of generation...
        hid_prob[2,:,:,gen_step]  = input_hid_prob #the hidden probability is the one in the input
        hid_states[2,:,:,gen_step]  = input_hid_prob
        c=1
        for rbm in reversed(dbn.rbm_layers):
            if c==1:
              p_v, v = rbm.backward(input_hid_prob)
              layer_size = v.shape[1]
              hid_prob[2-c,:,:layer_size,gen_step]  = p_v #the hidden probability is the one in the input
              hid_states[2-c,:,:layer_size,gen_step]  = v

            else:
              if c<len(dbn.rbm_layers):
                p_v, v = rbm.backward(v)
                layer_size = v.shape[1]
                hid_prob[2-c,:,:layer_size,gen_step]  = p_v #the hidden probability is the one in the input
                hid_states[2-c,:,:layer_size,gen_step]  = v
              else:
                v, p_v = rbm.backward(v) #passo la probabilità (che in questo caso è v) dopo
                layer_size = v.shape[1]
                vis_prob[:,:,gen_step]  = v #the hidden probability is the one in the input
                vis_states[:,:,gen_step]  = v
            c=c+1
      else:
            for rbm in dbn.rbm_layers:
              p_v, v = rbm(v)
            hid_prob[2,:,:,gen_step]  = p_v #the hidden probability is the one in the input
            hid_states[2,:,:,gen_step]  = v

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
              

    result_dict = dict(); 
    result_dict['hid_states'] = hid_states
    result_dict['vis_states'] = vis_states
    result_dict['hid_prob'] = hid_prob
    result_dict['vis_prob'] = vis_prob


    return result_dict


class Intersection_analysis_ZAMBRA:
    def __init__(self, model, top_k_Hidden=100, nr_steps=100):
        self.model = model
        self.top_k_Hidden = top_k_Hidden
        self.nr_steps = nr_steps
        
    def do_intersection_analysis(self):

      for dig in range(self.model.Num_classes):
        g_H = label_biasing_ZAMBRA(self.model, on_digits=dig, topk = -1)
        if dig == 0:
            hid_bias = g_H
        else:
            hid_bias = torch.hstack((hid_bias,g_H))

      vettore_indici_allDigits_biasing = torch.empty((0),device= self.model.DEVICE)

      for digit in range(self.model.Num_classes): #per ogni digit
        hid_vec_B = hid_bias[:,digit] #questo è l'hidden state ottenuto con il label biasing di un certo digit
        top_values_biasing, top_idxs_biasing = torch.topk(hid_vec_B, self.top_k_Hidden) #qui e la linea sotto  trovo i top p indici in termini di attività

        vettore_indici_allDigits_biasing = torch.cat((vettore_indici_allDigits_biasing,top_idxs_biasing),0) #concateno i top p indici di ciascun i digits in questo vettore

      unique_idxs_biasing,count_unique_idxs_biasing = torch.unique(vettore_indici_allDigits_biasing,return_counts=True) #degli indici trovati prendo solo quelli non ripetuti


      digit_digit_common_elements_count_biasing = torch.zeros((self.model.Num_classes,self.model.Num_classes))

      self.unique_H_idxs_biasing = unique_idxs_biasing

      result_dict_biasing ={}


      #itero per ogni digit per calcolare le entrate delle matrici 10 x 10
      for row in range(self.model.Num_classes): 
        for col in range(self.model.Num_classes):

          common_el_idxs_biasing = torch.empty((0),device= self.model.DEVICE)

          counter_biasing = 0
          for id in unique_idxs_biasing: #per ogni indice unico del biasing di ogni digit
            digits_found = torch.floor(torch.nonzero(vettore_indici_allDigits_biasing==id)/self.top_k_Hidden)
            #nella linea precedente torch.nonzero(vettore_indici_allDigits_biasing==id) trova le posizioni nell'array vettore_indici_allDigits_biasing
            #che ospitano l'unità ID. ora, essendo che vettore_indici_allDigits_biasing contiene le prime 100 unità più attive di ciascun digit, se divido gli indici per 100
            #trovo per quali digit l'unità ID era attiva
            if torch.any(digits_found==row) and torch.any(digits_found==col): #se i digits trovati ospitano sia il digit riga che quello colonna...
                common_el_idxs_biasing = torch.hstack((common_el_idxs_biasing,id)) #aggiungi ID al vettore di ID che verranno usati per fare biasing
                counter_biasing += 1

          result_dict_biasing[str(row)+','+str(col)] = common_el_idxs_biasing
          digit_digit_common_elements_count_biasing[row,col] = counter_biasing


      self.result_dict_biasing = result_dict_biasing

      print(digit_digit_common_elements_count_biasing)
      #lbl_bias_freqV = digit_digit_common_elements_count_biasing.view(100)/torch.sum(digit_digit_common_elements_count_biasing.view(100))

      return digit_digit_common_elements_count_biasing
    
    def generate_chimera_lbl_biasing(self,VGG_cl, elements_of_interest = [8,2], temperature=1, nr_of_examples = 1000, plot=0, entropy_correction=0):
      b_vec =torch.zeros(nr_of_examples,self.model.top_layer_size) #this 2000 seems the layersize hardcoded
      if not(elements_of_interest =='rand'):
        dictionary_key = str(elements_of_interest[0])+','+str(elements_of_interest[1])
        b_vec[:,self.result_dict_biasing[dictionary_key].long()]=1

      else: #write 'rand' in elements of interest
        for i in range(nr_of_examples):
          n1 = random.randint(0, self.model.Num_classes-1)
          n2 = random.randint(0, self.model.Num_classes-1)
          dictionary_key = str(n1)+','+str(n2)
          b_vec[i,self.result_dict_biasing[dictionary_key].long()]=1

      b_vec = torch.transpose(b_vec,0,1)
      #b_vec = torch.unsqueeze(b_vec,0)
      d = generate_from_hidden_ZAMBRA(self.model, b_vec, nr_gen_steps=self.nr_steps)
      
      d = Classifier_accuracy(d, VGG_cl, self.model, plot=plot, Thresholding_entropy=entropy_correction)
      df_average,df_sem, Transition_matrix_rowNorm = classification_metrics_ZAMBRA(d,self.model, Plot=plot, Ian=1)
      
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
          df_average,df_sem, Transition_matrix_rowNorm = classification_metrics_ZAMBRA(d,model,Plot=0,dS=50,Ian=1)
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
    LineW = 4
    Mk = 's'
    Mk_sz = 12
    Cp_sz = 12
    Err_bar_sz = 4
    Scritte_sz = 50

    # Define the two lists of four numbers
    MNIST = np.array(Results_dict['MNIST'][sel_key])
    CelebA =  np.array(Results_dict['CelebA_BW'][sel_key])

    # Define the two lists of four SEMs
    if sel_key=='Nr_visited_states_MEAN':
      MNIST_sem =  np.array(Results_dict['MNIST']['Nr_visited_states_SEM'])
      CelebA_sem =  np.array(Results_dict['CelebA_BW']['Nr_visited_states_SEM'])
      # Create a list of the x-axis labels
      x_labels = ['LB', 'C_2LB', 'C_int']
      x_lab = 'Generation method'
      y_lab = 'Number of states'
      y_r = [1,5]
    else:
      # Create a list of the x-axis labels
      x_labels = ['V', 'H1', 'H2', 'H3']
      x_lab = 'Layer'
      y_lab = 'Accuracy'
      y_r = [0.5,1]

    # Create a new figure and axis object
    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot the first line
    line1, = ax.plot(x_labels, MNIST, color='blue', label='MNIST', linewidth=LineW, marker=Mk, markersize=Mk_sz)
    # Plot the second line
    line2, = ax.plot(x_labels, CelebA, color='red', label='CelebA', linewidth=LineW, marker=Mk, markersize=Mk_sz)
    if  sel_key=='Nr_visited_states_MEAN':
      # Add error bars to the first line
      ax.errorbar(x_labels, MNIST, yerr=MNIST_sem, fmt='none', ecolor='blue', capsize=Cp_sz, elinewidth=Err_bar_sz)
      # Add error bars to the second line
      ax.errorbar(x_labels, CelebA, yerr=CelebA_sem, fmt='none', ecolor='red', capsize=Cp_sz,  elinewidth=Err_bar_sz)

    # Set the x-axis label
    ax.set_xlabel(x_lab, fontsize=Scritte_sz)

    # Set the y-axis label
    ax.set_ylabel(y_lab, fontsize=Scritte_sz)

    # Set the font size of all the text in the plot
    plt.rc('font', size=Scritte_sz)

    # Set the y-axis range
    ax.set_ylim(y_r)

    # Set the legend position and font size
    ax.legend(handles=[line1, line2], loc='upper center', bbox_to_anchor=(0.5, 0.3), fontsize=Scritte_sz)

    # Set the x-axis tick font size
    ax.tick_params(axis='x', labelsize=Scritte_sz)

    # Set the y-axis tick font size
    ax.tick_params(axis='y', labelsize=Scritte_sz)

    # Display the plot
    plt.show()

