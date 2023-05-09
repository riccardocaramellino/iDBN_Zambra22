from tqdm import tqdm
import os
import json
import numpy as np
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch
from dbns import *



def load_MNIST_data_ZAMBRA(CPARAMS,LPARAMS):
    #Riassume il caricamento dei dati del MNIST nella repository di Zambra, evitando problemi ed errori

    NUM_FEAT_MNIST = np.int32(784)
    DATASET_ID = CPARAMS['DATASET_ID']
    BATCH_SIZE = LPARAMS['BATCH_SIZE']

    mnist_train = datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose(
                        [transforms.ToTensor()] ))

    mnist_test = datasets.MNIST('../data', train=False, download=True,
                        transform=transforms.Compose(
                        [transforms.ToTensor()]) )


    train_loader = DataLoader(mnist_train, batch_size = BATCH_SIZE, shuffle = True)
    test_loader  = DataLoader(mnist_test, batch_size = BATCH_SIZE, shuffle = False)

    num_batches = mnist_train.__len__() // BATCH_SIZE
    train_data = torch.empty(num_batches, BATCH_SIZE, NUM_FEAT_MNIST)
    train_labels = torch.empty(num_batches, BATCH_SIZE, 1)

    with tqdm(train_loader, unit = 'Batch') as tdata:
        
        for idx, (batch, labels) in enumerate(tdata):
            if idx==0:
              b_Size0 =  batch.shape[0]
            tdata.set_description(f'Train Batch {idx}\t')
            if idx<468:
              bsize = batch.shape[0]
              train_data[idx,:,:] = batch.reshape(bsize, -1).type(torch.float32)
              train_labels[idx,:,:] = labels.reshape(bsize, -1).type(torch.float32)
        #end
    #end

    num_batches = mnist_test.__len__() // BATCH_SIZE
    test_data = torch.empty(num_batches, BATCH_SIZE, NUM_FEAT_MNIST)
    test_labels = torch.empty(num_batches, BATCH_SIZE, 1)

    with tqdm(test_loader, unit = 'Batch') as tdata:
        
        for idx, (batch, labels) in enumerate(tdata):
            tdata.set_description(f'Test Batch {idx}\t')
            if idx<78:
              bsize = batch.shape[0]
              test_data[idx,:,:] = batch.reshape(bsize, -1).type(torch.float32)
              test_labels[idx,:,:] = labels.reshape(bsize, -1).type(torch.float32)
        #end
    #end
    train_dataset = {'data' : train_data, 'labels' : train_labels}
    test_dataset  = {'data' : test_data, 'labels' : test_labels}

    return train_dataset, test_dataset


def tool_loader_ZAMBRA(DEVICE):
  from google.colab import drive
  drive.mount('/content/gdrive')
  Zambra_folder_drive = '/content/gdrive/My Drive/ZAMBRA_DBN/'

  #load the various files necessary
  with open(os.path.join(Zambra_folder_drive, 'cparams.json'), 'r') as filestream:
    CPARAMS = json.load(filestream)
  filestream.close()

  DATASET_ID = CPARAMS['DATASET_ID']
  ALG_NAME = CPARAMS['ALG_NAME']
  MODEL_NAME = f'{ALG_NAME}DBN'

  READOUT        = CPARAMS['READOUT']
  RUNS           = CPARAMS['RUNS']
  LAYERS         = CPARAMS['LAYERS']
  NUM_DISCR      = CPARAMS['NUM_DISCR']
  INIT_SCHEME    = CPARAMS['INIT_SCHEME']

  with open(os.path.join(Zambra_folder_drive, f'lparams-{DATASET_ID.lower()}.json'), 'r') as filestream:
    LPARAMS = json.load(filestream)
  filestream.close()
  
  EPOCHS         = LPARAMS['EPOCHS']
  INIT_MOMENTUM  = LPARAMS['INIT_MOMENTUM']
  FINAL_MOMENTUM = LPARAMS['FINAL_MOMENTUM']
  LEARNING_RATE  = LPARAMS['LEARNING_RATE']
  WEIGHT_PENALTY = LPARAMS['WEIGHT_PENALTY']

  train_dataset, test_dataset = load_MNIST_data_ZAMBRA(CPARAMS,LPARAMS)
  if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

  Training_ON = int(input('Do you want to train a DBN (Zambra 22 style) or do you want to load it? (1=yes, 0=no)'))
  
  if Training_ON == 1:
    Xtrain = train_dataset['data'].to(DEVICE)
    Ytrain = train_dataset['labels'].to(DEVICE)
    Xtest  = test_dataset['data'].to(DEVICE)
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
        if DATASET_ID == 'MNIST':
            model = [
                {'W' : 0.01 * torch.nn.init.normal_(torch.empty(784, 500), mean = 0, std = 1),  
                'a' : torch.zeros((1, 784)),  
                'b' : torch.zeros((1, 500))},
                {'W' : 0.01 * torch.nn.init.normal_(torch.empty(500, 500), mean = 0, std = 1),  
                'a' : torch.zeros((1, 500)),  
                'b' : torch.zeros((1, 500))},
                {'W' : 0.01 * torch.nn.init.normal_(torch.empty(500, 2000), mean = 0, std = 1), 
                'a' : torch.zeros((1, 500)),  
                'b' : torch.zeros((1, 2000))}
            ]
            
        elif DATASET_ID == 'SZ':
            model = [
                {'W' : 0.1 * torch.nn.init.normal_(0, 1, (900, 80)), 
                'a' : torch.zeros((1, 900)), 
                'b' : torch.zeros((1, 80))},
                {'W' : 0.1 * torch.nn.init.normal_(0, 1, (80, 400)), 
                'a' : torch.zeros((1, 80)),  
                'b' : torch.zeros((1, 400))}
            ]
        #end
        
        
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
        name = dbn.get_name()
        if run is not None:
            name += f'_run{run}'
        #end
        torch.save(dbn.to(torch.device('cpu')),
                    open(os.path.join(Zambra_folder_drive, f'{name}.pkl'), 'wb'))
    #end
  else:
    dbn = torch.load(os.path.join(Zambra_folder_drive, 'dbn_iterative_normal_MNIST_run0.pkl'))
  
  return dbn,train_dataset, test_dataset


def compute_inverseW_for_lblBiasing_ZAMBRA(model,train_dataset,DEVICE):

    lbls = train_dataset['labels'].view(-1)
    Num_classes=10
    L = torch.zeros(Num_classes,lbls.shape[0], device = DEVICE)

    c=0
    for lbl in lbls:
        L[int(lbl),c]=1
        c=c+1
    p_v, v = model(train_dataset['data'].cuda(), only_forward = True)
    V_lin = v.view(468*128, 2000)
    #I compute the inverse of the weight matrix of the linear classifier. weights_inv has shape (model.Num_classes x Hidden layer size (10 x 1000))
    weights_inv = torch.transpose(torch.matmul(torch.transpose(V_lin,0,1), torch.linalg.pinv(L)), 0, 1)

    model.weights_inv = weights_inv

    return weights_inv

def label_biasing_ZAMBRA(model, DEVICE, on_digits=1, topk = 149):

        # aim of this function is to implement the label biasing procedure described in
        # https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00515/full
        

        Num_classes=10
        # Now i set the label vector from which i will obtain the hidden layer of interest 
        Biasing_vec = torch.zeros (Num_classes,1, device = DEVICE)
        Biasing_vec[on_digits] = 1

        #I compute the biased hidden vector as the matmul of the trasposed weights_inv and the biasing vec. gen_hidden will have size (Hidden layer size x 1)
        gen_hidden= torch.matmul(torch.transpose(model.weights_inv,0,1), Biasing_vec)

        if topk>-1: #ATTENZIONE: label biasing con più di una label attiva (e.g. on_digits=[4,6]) funziona UNICAMENTE con topk>-1 (i.e. attivando le top k unità piu attive e silenziando le altre)
        #In caso contrario da errore CUDA non meglio specificato
          H = torch.zeros_like(gen_hidden, device = DEVICE) #crea un array vuoto della forma di gen_hidden
          for c in range(gen_hidden.shape[1]): # per ciascun esempio di gen_hidden...
            top_indices = torch.topk(gen_hidden[:,c], k=topk).indices # computa gli indici più attivati
            H[top_indices,c] = 1 #setta gli indici più attivi a 1
          gen_hidden = H # gen_hidden ha ora valori binari (1 e 0)



        return gen_hidden

