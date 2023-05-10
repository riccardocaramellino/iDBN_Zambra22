from tqdm import tqdm
import os
import json
import numpy as np
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch
from dbns import *
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import VGG_MNIST
import methods
from VGG_MNIST import *
from methods import *
from google.colab import files
from itertools import combinations




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

  Load_DBN_yn = int(input('Do you want to load a iDBN (Zambra 22 style) or do you want to train it? (1=yes, 0=no)'))
  
  if Load_DBN_yn == 0:
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
        dbn.Num_classes = 10
        dbn.DEVICE = DEVICE
        compute_inverseW_for_lblBiasing_ZAMBRA(dbn,train_dataset)


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


def compute_inverseW_for_lblBiasing_ZAMBRA(model,train_dataset):

    lbls = train_dataset['labels'].view(-1)
    Num_classes=10
    L = torch.zeros(Num_classes,lbls.shape[0], device = model.DEVICE)

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

def label_biasing_ZAMBRA(model, on_digits=1, topk = 149):

        # aim of this function is to implement the label biasing procedure described in
        # https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00515/full
        

        Num_classes=10
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

    hid_prob = torch.zeros(len(dbn.rbm_layers),numcases,hidden_layer_size, nr_gen_steps, device=dbn.DEVICE)
    hid_states = torch.zeros(len(dbn.rbm_layers), numcases,hidden_layer_size, nr_gen_steps, device=dbn.DEVICE)
    vis_prob = torch.zeros(numcases, 784, nr_gen_steps, device=dbn.DEVICE)
    vis_states = torch.zeros(numcases ,784, nr_gen_steps, device=dbn.DEVICE)


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
      #vettore_indici_allDigits_hidAvg = torch.empty((0),device= self.model.DEVICE)

      for digit in range(self.model.Num_classes): #per ogni digit
        hid_vec_B = hid_bias[:,digit] #questo è l'hidden state ottenuto con il label biasing di un certo digit
        top_values_biasing, top_idxs_biasing = torch.topk(hid_vec_B, self.top_k_Hidden) #qui e la linea sotto  trovo i top p indici in termini di attività

        vettore_indici_allDigits_biasing = torch.cat((vettore_indici_allDigits_biasing,top_idxs_biasing),0) #concateno i top p indici di ciascun i digits in questo vettore

      unique_idxs_biasing,count_unique_idxs_biasing = torch.unique(vettore_indici_allDigits_biasing,return_counts=True) #degli indici trovati prendo solo quelli non ripetuti

      #common_el_idxs_hidAvg = torch.empty((0),device= self.model.DEVICE)
      #common_el_idxs_biasing = torch.empty((0),device= self.model.DEVICE)

      digit_digit_common_elements_count_biasing = torch.zeros((10,10))

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
      lbl_bias_freqV = digit_digit_common_elements_count_biasing.view(100)/torch.sum(digit_digit_common_elements_count_biasing.view(100))

      return digit_digit_common_elements_count_biasing
    def generate_chimera_lbl_biasing(self,VGG_cl, elements_of_interest = [8,2], temperature=1, nr_of_examples = 1000, plot=0, entropy_correction=0):
      b_vec =torch.zeros(nr_of_examples,2000) #this 2000 seems the layersize hardcoded
      if not(elements_of_interest =='rand'):
        dictionary_key = str(elements_of_interest[0])+','+str(elements_of_interest[1])
        b_vec[:,self.result_dict_biasing[dictionary_key].long()]=1

      else: #write 'rand' in elements of interest
        for i in range(nr_of_examples):
          n1 = random.randint(0, 9)
          n2 = random.randint(0, 9)
          dictionary_key = str(n1)+','+str(n2)
          b_vec[i,self.result_dict_biasing[dictionary_key].long()]=1

      b_vec = torch.transpose(b_vec,0,1)
      #b_vec = torch.unsqueeze(b_vec,0)
      d = generate_from_hidden_ZAMBRA(self.model, b_vec, nr_gen_steps=self.nr_steps)
      
      d = Classifier_accuracy(d, VGG_cl, self.model, plot=plot, Thresholding_entropy=entropy_correction)
      df_average,df_sem, Transition_matrix_rowNorm = classification_metrics(d,self.model, Plot=plot, Ian=1)
      
      if nr_of_examples < 16:
        Plot_example_generated(d, self.model ,row_step = 10, dS=20, custom_steps = True, Show_classification = False)

      
      return d, df_average,df_sem, Transition_matrix_rowNorm
    
def Chimeras_nr_visited_states_ZAMBRA(model, VGG_cl, Ian =[], topk=149, apprx=1,plot=1,compute_new=1, nr_sample_generated =100, entropy_correction=[], lS=20):
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
      Non_digit_mat  = np.zeros((n_digits, n_digits))
      Non_digit_err  = np.zeros((n_digits, n_digits))

      if Ian!=[]:
        for row in range(n_digits):
          for col in range(row,n_digits):
            d, df_average,df_sem, Transition_matrix_rowNorm = Ian.generate_chimera_lbl_biasing(VGG_cl,elements_of_interest = [row,col], nr_of_examples = nr_sample_generated, temperature = 1, plot=0, entropy_correction= entropy_correction)
            Vis_states_mat[row,col]=df_average.Nr_visited_states[0]
            Vis_states_err[row,col]=df_sem.Nr_visited_states[0]
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
          Non_digit_mat[combination[0],combination[1]] = df_average['Non-digit'][0]
          Non_digit_err[combination[0],combination[1]] = df_sem['Non-digit'][0]


      save_mat_xlsx(Vis_states_mat, filename=fN)
      save_mat_xlsx(Vis_states_err, filename=fNerr)
      save_mat_xlsx(Non_digit_mat, filename=fN_NDST)
      save_mat_xlsx(Non_digit_err, filename=fNerr_NDST)

    else: #load already computed Vis_states_mat
      Vis_states_mat = pd.read_excel(fN)
      Non_digit_mat = pd.read_excel(fN_NDST)
      # Convert the DataFrame to a NumPy array
      Vis_states_mat = Vis_states_mat.values
      Non_digit_mat = Non_digit_mat.values

      Vis_states_err = pd.read_excel(fNerr)
      Non_digit_err = pd.read_excel(fNerr_NDST)
      # Convert the DataFrame to a NumPy array
      Vis_states_err = Vis_states_err.values
      Non_digit_err = Non_digit_err.values

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

      #ax.set_xticklabels(T_mat_labels)
      ax.tick_params(axis='both', labelsize=lS)

      plt.xlabel('Digit', fontsize = 25) # x-axis label with fontsize 15
      plt.ylabel('Digit', fontsize = 25) # y-axis label with fontsize 15
      #cbar = plt.gcf().colorbar(ax.collections[0], location='left', shrink=0.82)
      cbar = ax.collections[0].colorbar
      cbar.ax.tick_params(labelsize=lS)
      plt.show()

    return Vis_states_mat, Vis_states_err,Non_digit_mat,Non_digit_err







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