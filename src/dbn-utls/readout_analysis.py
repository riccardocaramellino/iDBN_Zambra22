import random
from tqdm import tqdm
import torch
from torchvision import datasets,transforms
import pickle
import os
import json
from sklearn.metrics import accuracy_score
import Study_generativity
from Study_generativity import *

def readout_V_to_Hlast(dbn,train_dataset,test_dataset, DEVICE='cuda', existing_classifier_list = []):
  classifier_list = []
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

  #upper = int(Xtrain.shape[0]*(4/5))
  upper = Xtrain.shape[0]
  if len(existing_classifier_list) == 0:
    readout_acc,classifier = dbn.rbm_layers[1].get_readout(Xtrain[:upper, :, :], Xtest, Ytrain[:upper, :, :], Ytest)
  else:
    n_feat = Xtrain.shape[-1]
    x_test  = Xtest.cpu().numpy().reshape(-1, n_feat)
    y_test  = Ytest.cpu().numpy().flatten()
    classifier = existing_classifier_list[0]
    y_pred = classifier.predict(x_test)
    readout_acc = accuracy_score(y_test, y_pred)

  classifier_list.append(classifier)
  print(f'Readout accuracy = {readout_acc*100:.2f}')
  readout_acc_V.append(readout_acc)


  for rbm_idx,rbm in enumerate(dbn.rbm_layers):
      Xtrain = Xtrain.to(DEVICE)
      Xtest  = Xtest.to(DEVICE)
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
      print(rbm_idx)
      #if rbm_idx==2:
      print('eccomi')
      #upper = int(_Xtrain.shape[0]*(4/5))
      upper = _Xtrain.shape[0]

      if len(existing_classifier_list) == 0:
        readout_acc, classifier = rbm.get_readout(_Xtrain[:upper, :, :], _Xtest, Ytrain[:upper, :, :], Ytest)
      else:
        n_feat = _Xtrain.shape[-1]
        x_test  = _Xtest.cpu().numpy().reshape(-1, n_feat)
        y_test  = Ytest.cpu().numpy().flatten()
        classifier = existing_classifier_list[rbm_idx+1]
        y_pred = classifier.predict(x_test)
        readout_acc = accuracy_score(y_test, y_pred)

      classifier_list.append(classifier)
      print(f'Readout accuracy = {readout_acc*100:.2f}')
      #end
      readout_acc_V.append(readout_acc)

      Xtrain = _Xtrain.clone()
      Xtest  = _Xtest.clone()
  return readout_acc_V, classifier_list


def get_relative_freq(valore, hist, bin_edges,numero_bin=20):
    # Trova il bin in cui si trova il valore
    indice_bin = np.digitize(valore, bin_edges)

    # Controlla se l'indice è fuori dai limiti
    if 1 <= indice_bin <= numero_bin:
        frequenza_relativa = hist[indice_bin - 1]
        return frequenza_relativa
    else:
        return 0.0  # Il valore è al di fuori dei bin

def sampling_gen_examples(results, prob_distr, cumulative_sum,desired_len_array = 9984):
  random_numbers = np.random.rand(desired_len_array*10)
  index_selected_samples = []
  c=0
  while len(index_selected_samples)<desired_len_array:
     r = random_numbers[c]
     delta = cumulative_sum - r
     delta[delta<0] = np.inf
     index_p = (delta).argmin()
     indexes_suitable_imgs = torch.nonzero(results == prob_distr[index_p]).squeeze()
     if indexes_suitable_imgs.numel() > 1:
      random_index = random.choice(indexes_suitable_imgs.tolist())
      if not(random_index in index_selected_samples):
        index_selected_samples.append(random_index)
     elif indexes_suitable_imgs.numel() == 1:
      if not(random_index in index_selected_samples):
        index_selected_samples.append(int(indexes_suitable_imgs))
     c=c+1
  return index_selected_samples

def load_existing_retrainDS(trainfile_path, testfile_path, nr_batches_retraining):

  train_dataset = dict(np.load(trainfile_path))
  test_dataset = dict(np.load(testfile_path))
  # Convert the numpy arrays to torch tensors
  for key in train_dataset:
      train_dataset[key] = torch.from_numpy(train_dataset[key])
      test_dataset[key]= torch.from_numpy(test_dataset[key])

  train_data_retraining_ds = train_dataset['data'][:nr_batches_retraining,:,:]
  train_labels_retraining_ds = train_dataset['labels'][:nr_batches_retraining,:,:]
  train_dataset_retraining_ds = {'data': train_data_retraining_ds, 'labels': train_labels_retraining_ds}
  test_dataset_retraining_ds = test_dataset
  return train_dataset_retraining_ds, test_dataset_retraining_ds

def random_selection_withinBatch(batch_size=128):
  # Creazione di una lista di tutti gli indici possibili da 0 a 127
  all_indices = list(range(batch_size))
  # Estrai 64 indici casuali dalla lista
  random.shuffle(all_indices)
  selected_indices = all_indices[:batch_size//2]
  # Gli altri 64 indici sono quelli rimanenti
  remaining_indices = all_indices[batch_size//2:]
  return selected_indices, remaining_indices

def mixing_data_wihin_batch(half_MNIST,half_retraining_ds):
  batch_size = half_MNIST.shape[1] 
  nr_batches = half_MNIST.shape[0]
  mix_retraining_ds_MNIST = torch.zeros((nr_batches*2,batch_size,half_MNIST.shape[2]))

  randomly_selected_batches_MNIST = list(range(nr_batches))
  random.shuffle(randomly_selected_batches_MNIST)

  randomly_selected_batches_retrainDS = list(range(nr_batches))
  random.shuffle(randomly_selected_batches_retrainDS)

  #print([randomly_selected_batches_MNIST[i]== randomly_selected_batches_retrainDS[i] for i in range(len(randomly_selected_batches_retrainDS))])

  for idx in range(nr_batches):

      selected_indices_MNIST, remaining_indices_MNIST = random_selection_withinBatch(batch_size)
      selected_indices_retrainDS, remaining_indices_retrainDS = random_selection_withinBatch(batch_size)
      
      # if idx==0:
      #   print([selected_indices_MNIST[i]== selected_indices_retrainDS[i] for i in range(len(selected_indices_retrainDS))])
      #   print([selected_indices_MNIST[i]== remaining_indices_MNIST[i] for i in range(len(remaining_indices_MNIST))])

      MNIST_examples = half_MNIST[randomly_selected_batches_MNIST[idx],selected_indices_MNIST,:]
      retrainDS_examples = half_retraining_ds[randomly_selected_batches_retrainDS[idx],selected_indices_retrainDS,:]

      mix_batch1 = torch.cat((MNIST_examples, retrainDS_examples), dim=0)
      permuted_indices = torch.randperm(batch_size)
      # Use the permutation to shuffle the examples of the dataset
      mix_batch1 = mix_batch1[permuted_indices]
      mix_retraining_ds_MNIST[idx,:,:] = mix_batch1


      MNIST_examples2 = half_MNIST[randomly_selected_batches_MNIST[idx],remaining_indices_MNIST,:]
      retrainDS_examples2 = half_retraining_ds[randomly_selected_batches_retrainDS[idx],remaining_indices_retrainDS,:]

      mix_batch2 = torch.cat((MNIST_examples2, retrainDS_examples2), dim=0)
      permuted_indices = torch.randperm(batch_size)
      # Use the permutation to shuffle the examples of the dataset
      mix_batch2 = mix_batch2[permuted_indices]
      mix_retraining_ds_MNIST[idx+(nr_batches//2),:,:] = mix_batch2

  return mix_retraining_ds_MNIST
  

def get_retraining_data(MNIST_train_dataset, train_dataset_retraining_ds = {}, dbn=[], classifier=[], n_steps_generation = 10, ds_type = 'EMNIST', half_MNIST_gen=True, Type_gen = 'chimeras', H_type = 'det', selection_gen = False, correction_type = 'frequency'):
  #Type_gen = 'chimeras'/'lbl_bias'/'mix'
  #NOTA: il labelling dell'EMNIST by class ha 62 labels: le cifre (0-9), lettere MAUSCOLE (10-36), lettere MINUSCOLE(38-62)
  #20,000 uppercase letters from the first 10 EMNIST classes.
  coeff = 1
  nr_batches_retraining = round(20000/128) 
  half_batches = round(nr_batches_retraining/2)
  half_ds_size = half_batches*128 #i.e. 9984
  

  if selection_gen == True and half_MNIST_gen==True:
    coeff = 2 #moltiplicatore. Un tempo stava a 2
    vectors = []
    for batch in MNIST_train_dataset['data']: #one batch at a time
        vectors.append(torch.mean(batch,axis = 1))
    avg_pixels_active_TrainMNIST = torch.cat(vectors) #This is the distribution of avg pixels active in the MNIST train dataset


  def decrease_labels_by_10(data, sorted_list):
    image, label = data
    return image, sorted_list.index(label)

  #load  EMNIST byclass data
  if not(bool(train_dataset_retraining_ds)):
    Zambra_folder_drive = '/content/gdrive/My Drive/ZAMBRA_DBN/'
    test_filename = 'test_dataset_'+ds_type+'.npz'
    train_filename = 'train_dataset_'+ds_type+'.npz'
    trainfile_path= os.path.join(Zambra_folder_drive,'dataset_dicts',train_filename)
    testfile_path = os.path.join(Zambra_folder_drive,'dataset_dicts',test_filename) 
    if os.path.exists(trainfile_path):
      train_dataset_retraining_ds, test_dataset_retraining_ds = load_existing_retrainDS(trainfile_path, testfile_path, nr_batches_retraining)
    else:
      transform =transforms.Compose([transforms.ToTensor()])
      if ds_type == 'EMNIST':
          data_train_retraining_ds = datasets.EMNIST('../data', train=True,split = 'byclass', download=True, transform=transform)
          data_test_retraining_ds = datasets.EMNIST('../data', train=False,split = 'byclass', download=True, transform=transform)
          #target_classes = list(range(10, 20)) #i.e. the first 10 capital letter classes
          target_classes = [22,32,26,16,30,11,20,10,23,25] #migliori dritte: [22,32,26,16,30,11,20,10,23,25], medi[17,18,19,20,21,22,23,24,25,26]
          sorted_list = sorted(target_classes)
          data_train_retraining_ds = [item for item in data_train_retraining_ds if item[1] in target_classes]
          data_test_retraining_ds = [item for item in data_test_retraining_ds if item[1] in target_classes]
          #i relabel data from 10-19 to 0-9
          data_train_retraining_ds = [decrease_labels_by_10(item,sorted_list) for item in data_train_retraining_ds]
          data_test_retraining_ds = [decrease_labels_by_10(item,sorted_list) for item in data_test_retraining_ds]
          #questi loop sono per raddrizzare le lettere
          data_train_retraining_L = []
          for item in data_train_retraining_ds:
            image = item[0].view(28, 28)
            image = torch.rot90(image, k=-1)
            image = torch.flip(image, [1])
            data_train_retraining_L.append((image,item[1]))

          data_test_retraining_L = []
          for item in data_test_retraining_ds:
            image= item[0].view(28, 28)
            image = torch.rot90(image, k=-1)
            image = torch.flip(image, [1])
            data_test_retraining_L.append((image,item[1]))

          data_test_retraining_ds = data_test_retraining_L
          data_train_retraining_ds = data_train_retraining_L
      elif ds_type == 'fMNIST':
          data_train_retraining_ds = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
          data_test_retraining_ds = datasets.FashionMNIST('../data', train=False, download=True, transform=transform)

      train_data_retraining_ds, train_labels_retraining_ds = data_and_labels(data_train_retraining_ds, BATCH_SIZE=128,NUM_FEAT=np.int32(28*28),DATASET_ID='MNIST',n_cols_labels=1)
      test_data_retraining_ds, test_labels_retraining_ds = data_and_labels(data_test_retraining_ds, BATCH_SIZE=128,NUM_FEAT=np.int32(28*28),DATASET_ID='MNIST',n_cols_labels=1)
    
      #i select just 20000 examples (19968 per l'esattezza)
      train_data_retraining_ds = train_data_retraining_ds[:nr_batches_retraining,:,:]
      train_labels_retraining_ds = train_labels_retraining_ds[:nr_batches_retraining,:,:]
      train_dataset_retraining_ds = {'data': train_data_retraining_ds, 'labels': train_labels_retraining_ds}
      test_dataset_retraining_ds = {'data': test_data_retraining_ds, 'labels': test_labels_retraining_ds}

    if dbn==[]:
        return train_dataset_retraining_ds,test_dataset_retraining_ds
    
  if not(half_MNIST_gen):
     half_MNIST = MNIST_train_dataset['data'][:half_batches,:,:].to('cuda')
  else:
    compute_inverseW_for_lblBiasing_ZAMBRA(dbn,MNIST_train_dataset)
    if H_type == 'det':
      for dig in range(dbn.Num_classes): #at the end of this loop, you have one example of label biasing per class
          g_H = label_biasing_ZAMBRA(dbn, on_digits=dig, topk = -1)
          if dig == 0:
              g_H0to9 = g_H
          else:
              g_H0to9 = torch.hstack((g_H0to9,g_H)) #final size: [1000, 10]
      n_samples = math.ceil(10000*coeff/(10*n_steps_generation))
      gen_hidden_100rep = g_H0to9.repeat(1,n_samples)
      
    #NON FUNZIONA:
    #noise = torch.normal(mean=0.0, std=1, size=(1000, n_samples*10))
    #gen_hidden_100rep = gen_hidden_100rep + noise
    elif H_type == 'stoch':
      original_W = dbn.weights_inv

      for it in range(n_samples):
        #SD = random.uniform(0, float(torch.std(original_W)))
        SD = float(torch.std(original_W))
        dbn.weights_inv = original_W + torch.normal(mean=0.0, std=SD, size=dbn.weights_inv.shape)

        for dig in range(dbn.Num_classes):
          g_H = label_biasing_ZAMBRA(dbn, on_digits=dig, topk = -1)
          if dig == 0:
            g_H0to9 = g_H
          else:
            g_H0to9 = torch.hstack((g_H0to9,g_H))
        gen_hidden_100rep[:,((it+1)*10)-10:((it+1)*10)] = g_H0to9

    else:
      tensor_size = (1000, n_samples * 10)
      # Generate the tensor of random values from uniform distribution
      gen_hidden_100rep = torch.rand(tensor_size)

    VStack_labels=torch.tensor(range(dbn.Num_classes), device = 'cuda')
    VStack_labels=VStack_labels.repeat(n_samples)
    dict_DBN_lBias_classic = generate_from_hidden_ZAMBRA(dbn, gen_hidden_100rep, nr_gen_steps=n_steps_generation)
    
    if Type_gen == 'lbl_bias':
      Vis_states = dict_DBN_lBias_classic['vis_states'].permute(0, 2, 1)
      Vis_states = Vis_states.reshape(Vis_states.shape[0]*Vis_states.shape[1],Vis_states.shape[2]) #Vis_states.shape[2]=784
      indices = torch.randperm(Vis_states.size(0))[:math.ceil(half_ds_size*coeff)]
      # Sample the rows using the generated indices
      sampled_data = Vis_states[indices]
       
    else:
       Mean, _ = Perc_H_act(dbn, VStack_labels, gen_data_dictionary=dict_DBN_lBias_classic, dS = 50, l_sz = 5, layer_of_interest=2)
       k = int((torch.mean(Mean, axis=0)[0]*dbn.top_layer_size)/100)
       Ian = Intersection_analysis_ZAMBRA(dbn, top_k_Hidden=k,nr_steps=n_steps_generation)
       digit_digit_common_elements_count_biasing = Ian.do_intersection_analysis()
       n_samples = math.ceil(10000*coeff/(45*n_steps_generation))
       c=0
       for row in range(10):
          for col in range(row+1,10): #45 combinations(upper diagonal)
            d, df_average,df_sem, Transition_matrix_rowNorm = Ian.generate_chimera_lbl_biasing(classifier,elements_of_interest = [row,col], nr_of_examples = n_samples, temperature = 1, plot=0, entropy_correction=[])
            if c==0:
                Chim_gen_ds = d['vis_states'][:,:,:n_steps_generation]
            else:
                Chim_gen_ds = torch.cat((Chim_gen_ds, d['vis_states'][:,:,:n_steps_generation]), dim=0)
            c=c+1

       Vis_states_chimera = Chim_gen_ds.permute(0, 2, 1)
       Vis_states_chimera = Vis_states_chimera.reshape(Vis_states_chimera.shape[0]*Vis_states_chimera.shape[1],Vis_states_chimera.shape[2]) #Vis_states.shape[2]=784
       indices = torch.randperm(Vis_states_chimera.size(0))[:math.ceil(half_ds_size*coeff)]
       # Sample the rows using the generated indices
       sampled_data = Vis_states_chimera[indices]
    
    if selection_gen == True and half_MNIST_gen==True:
        avg_activity_sampled_data =  torch.mean(sampled_data,axis = 1)
        hist, bin_edges = np.histogram(avg_pixels_active_TrainMNIST, bins=20, density=True)
        sum_hist = np.sum(hist)
        prob_distr = hist/sum_hist
        cumulative_sum = np.cumsum(prob_distr)

        results = torch.zeros_like(avg_activity_sampled_data)

        for i in range(avg_activity_sampled_data.size(0)):
            value = avg_activity_sampled_data[i].item()
            results[i] = get_relative_freq(value, hist, bin_edges)
            if correction_type == 'sampling':
               results[i] = results[i]/sum_hist
        if correction_type == 'frequency':
          top_indices = torch.topk(results, k=half_ds_size).indices
        elif correction_type == 'sampling':
          top_indices = torch.tensor(sampling_gen_examples(results, prob_distr, cumulative_sum,desired_len_array = half_ds_size + 1000)) #200 è per evitare di andare sotto 9984
          top_indices = top_indices[:half_ds_size]
          number_of_unique_elements = len(torch.unique(top_indices))
          # Print in bold
          print(f"\033[1mNumber of unique elements: {number_of_unique_elements}\033[0m")
        else:
          top_indices = torch.tensor(np.where(results.cpu() != 0)[0])
          random_indices = torch.randperm(top_indices.size(0))
          top_indices = top_indices[random_indices[:half_ds_size]]

        sampled_data = sampled_data[top_indices]
        avg_activity_sampled_data_topK =  torch.mean(sampled_data,axis = 1)
        plt.figure()
        plt.hist(avg_pixels_active_TrainMNIST.cpu(), bins=20, color='blue', alpha=0.7,density=True, label='MNIST train set')  # You can adjust the number of bins as needed
        plt.hist(avg_activity_sampled_data.cpu(), bins=20, color='red', alpha=0.7,density=True, label='Generated data - no correction')
        plt.hist(avg_activity_sampled_data_topK.cpu(), bins=20, color='orange', alpha=0.7,density=True, label='Generated data - corrected')
        # Add labels and a title
        plt.xlabel('Average pixel activation')
        plt.ylabel('Relative frequency (%)')
        plt.legend()
        plt.show()

    half_MNIST = sampled_data.view(half_batches, 128, 784)

  half_retraining_ds = train_dataset_retraining_ds['data'][:half_batches,:,:].to('cuda')
  
  #mix_retraining_ds_MNIST = torch.cat((half_MNIST, half_retraining_ds), dim=0)
  mix_retraining_ds_MNIST=mixing_data_wihin_batch(half_MNIST,half_retraining_ds)

  # Generate a random permutation of indices
  permuted_indices = torch.randperm(nr_batches_retraining)
  # Use the permutation to shuffle the examples of the dataset
  mix_retraining_ds_MNIST = mix_retraining_ds_MNIST[permuted_indices]
  try:
    # Prova ad accedere alla variabile 'my_variable'.
    print(test_dataset_retraining_ds)
    return train_dataset_retraining_ds,test_dataset_retraining_ds,mix_retraining_ds_MNIST
  except NameError:
    # Gestisci l'eccezione se la variabile non esiste.
    print("La variabile 'my_variable' non esiste.")
    return mix_retraining_ds_MNIST
  

def get_ridge_classifiers(MNIST_Train_DS, MNIST_Test_DS, Force_relearning = True, last_layer_sz=1000):
  Zambra_folder_drive = '/content/gdrive/My Drive/ZAMBRA_DBN/'
  MNIST_rc_file= os.path.join(Zambra_folder_drive,'MNIST_ridge_classifiers'+str(last_layer_sz)+'.pkl')
  #EMNIST_rc_file= os.path.join(Zambra_folder_drive,'EMNIST_ridge_classifiers.pkl')
  print("\033[1m Make sure that your iDBN was trained only with MNIST for 100 epochs \033[0m")
  DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  dbn,train_dataset, test_dataset,classifier= tool_loader_ZAMBRA(DEVICE, only_data = False,last_layer_sz=last_layer_sz, Load_DBN_yn = 1)
  if not(os.path.exists(MNIST_rc_file)) or Force_relearning:
    readout_acc_V, MNIST_classifier_list = readout_V_to_Hlast(dbn,MNIST_Train_DS, MNIST_Test_DS)
    # Save the list of classifiers to a file
    with open(MNIST_rc_file, 'wb') as file:
        pickle.dump(MNIST_classifier_list, file)
  else:
    with open(MNIST_rc_file, 'rb') as file:
      MNIST_classifier_list = pickle.load(file)# Load the list of classifiers from the file

  # if not(os.path.exists(EMNIST_rc_file)) or Force_relearning:
  #   train_dataset_EMNIST,test_dataset_EMNIST, _ =get_retraining_data(train_dataset)
  #   Xtrain = train_dataset_EMNIST['data'].to(DEVICE)
  #   Xtest  = test_dataset_EMNIST['data'].to(DEVICE)
  #   Ytrain = train_dataset_EMNIST['labels'].to(DEVICE)
  #   Ytest  = test_dataset_EMNIST['labels'].to(DEVICE)
  #   DATASET_ID='MNIST'
  #   with open(os.path.join(Zambra_folder_drive, f'lparams-{DATASET_ID.lower()}.json'), 'r') as filestream:
  #       LPARAMS = json.load(filestream)
  #   with open(os.path.join(Zambra_folder_drive, 'cparams.json'), 'r') as filestream:
  #       CPARAMS = json.load(filestream)
  #   LPARAMS['EPOCHS']=100
  #   READOUT = CPARAMS['READOUT']
  #   NUM_DISCR = CPARAMS['NUM_DISCR']
  #   dbn.train(Xtrain, Xtest, Ytrain, Ytest, LPARAMS, readout = READOUT, num_discr = NUM_DISCR)
  #   readout_acc_V, EMNIST_classifier_list = readout_V_to_Hlast(dbn,train_dataset_EMNIST,test_dataset_EMNIST)
  #   # Save the list of classifiers to a file
  #   with open(EMNIST_rc_file, 'wb') as file:
  #       pickle.dump(EMNIST_classifier_list, file)
  # else:
  #   with open(EMNIST_rc_file, 'rb') as file:
  #     EMNIST_classifier_list = pickle.load(file)# Load the list of classifiers from the file
  #return MNIST_classifier_list, EMNIST_classifier_list
  return MNIST_classifier_list



def relearning(retrain_ds_type = 'EMNIST', mixing_type =[], n_steps_generation=10, new_retrain_data = False, selection_gen = False, correction_type = 'frequency', l_par = 5,  last_layer_sz = 1000, H_type='det'):
    DEVICE='cuda'
    dbn,MNISTtrain_ds, MNISTtest_ds,classifier= tool_loader_ZAMBRA(DEVICE, only_data = False,Load_DBN_yn = 1, last_layer_sz=last_layer_sz)
    mixing_type_options = ['origMNIST', 'lbl_bias', 'chimeras','[]']
    type_retrain = 'interleaved'
    type_mix = 'mix_'+mixing_type
    if mixing_type==[]:
        mixing_type_options_list = '\n'.join([f'{i}: {opt}' for i, opt in enumerate(mixing_type_options)]) # questa linea serve per creare il prompt di selezione del soggetto
        mixing_type_idx = int(input('Which mixing type for retraining?\n'+mixing_type_options_list))
        mixing_type = mixing_type_options[mixing_type_idx]
        type_mix = 'mix_'+mixing_type

    if mixing_type=='[]':
        mixing_type=[]
        type_retrain = 'sequential'
        type_mix = ''
    
    if mixing_type == 'origMNIST':
       half_MNIST_gen_option = False
    else:
       half_MNIST_gen_option = True

    Retrain_ds,Retrain_test_ds,mix_retrain_ds = get_retraining_data(MNISTtrain_ds,{},dbn, classifier,n_steps_generation = n_steps_generation,  ds_type = retrain_ds_type, half_MNIST_gen=half_MNIST_gen_option, Type_gen = mixing_type,H_type = H_type, selection_gen = selection_gen, correction_type = correction_type)
    MNIST_classifier_list= get_ridge_classifiers(MNISTtrain_ds, MNISTtest_ds,Force_relearning = False, last_layer_sz=last_layer_sz)

    
    dbn,train_dataset, test_dataset,_= tool_loader_ZAMBRA(DEVICE, only_data = False,last_layer_sz=last_layer_sz, Load_DBN_yn = 1)
    Zambra_folder_drive = '/content/gdrive/My Drive/ZAMBRA_DBN/'
    DATASET_ID='MNIST'

    with open(os.path.join(Zambra_folder_drive, f'lparams-{DATASET_ID.lower()}.json'), 'r') as filestream:
        LPARAMS = json.load(filestream)
    with open(os.path.join(Zambra_folder_drive, 'cparams.json'), 'r') as filestream:
        CPARAMS = json.load(filestream)

    if new_retrain_data == True and l_par==1:
       LPARAMS['EPOCHS']=1
       inner_loop_epochs = 5
    else:
       LPARAMS['EPOCHS']=5
       inner_loop_epochs = 1

    READOUT = CPARAMS['READOUT']
    NUM_DISCR = CPARAMS['NUM_DISCR']

    if mixing_type == []:
        Xtrain = Retrain_ds['data'].to(DEVICE)
    else:
       Xtrain = mix_retrain_ds.to(DEVICE)
    Xtest  = Retrain_test_ds['data'].to(DEVICE)
    Ytrain = Retrain_ds['labels'].to(DEVICE)
    Ytest  = Retrain_test_ds['labels'].to(DEVICE)
    nr_iter_training=10
    readout_acc_Seq_DIGITS = np.zeros((nr_iter_training+1,4))
    readout_acc_V_DIGITS,_ = readout_V_to_Hlast(dbn,train_dataset,test_dataset,existing_classifier_list = MNIST_classifier_list)
    readout_acc_Seq_DIGITS[0,:] = readout_acc_V_DIGITS

    readout_acc_Seq_RETRAINING_DS = np.zeros((nr_iter_training+1,4))
    readout_acc_V_LETTERS,_ = readout_V_to_Hlast(dbn,Retrain_ds,Retrain_test_ds)
    readout_acc_Seq_RETRAINING_DS[0,:] = readout_acc_V_LETTERS



    for iteration in range(nr_iter_training):
        for ep in range(inner_loop_epochs):
          if new_retrain_data == True:
            mix_retrain_ds = get_retraining_data(MNISTtrain_ds,Retrain_ds,dbn, classifier,n_steps_generation = n_steps_generation,  ds_type = retrain_ds_type, half_MNIST_gen=half_MNIST_gen_option, Type_gen = mixing_type,H_type =H_type, selection_gen = selection_gen, correction_type = correction_type)
            Xtrain = mix_retrain_ds.to(DEVICE)
            print(ep)
          dbn.train(Xtrain, Xtest, Ytrain, Ytest, LPARAMS, readout = READOUT, num_discr = NUM_DISCR)
        readout_acc_V_DIGITS,_ = readout_V_to_Hlast(dbn,train_dataset,test_dataset,existing_classifier_list = MNIST_classifier_list)
        #readout_acc_V_DIGITS,_ = readout_V_to_Hlast(dbn,train_dataset,test_dataset) retrain at every iteration on digits
        readout_acc_Seq_DIGITS[iteration+1,:] = readout_acc_V_DIGITS

        readout_acc_V_LETTERS,_ = readout_V_to_Hlast(dbn,Retrain_ds,Retrain_test_ds)
        readout_acc_Seq_RETRAINING_DS[iteration+1,:] = readout_acc_V_LETTERS
    # Creare un DataFrame da questi dati
    df_readout_RETRAINING_DS = pd.DataFrame(readout_acc_Seq_RETRAINING_DS)
    # Salva il DataFrame in un file Excel
    df_readout_RETRAINING_DS.to_excel('Readout_on_'+retrain_ds_type+'_retrain_on_'+retrain_ds_type+'_'+type_retrain+'_'+type_mix+'.xlsx', index=False, header=False)
    Readout_last_layer_RETRAINING_DS = df_readout_RETRAINING_DS.values[:, len(dbn.rbm_layers)]

    # Creare un DataFrame da questi dati
    df_readout_MNIST = pd.DataFrame(readout_acc_Seq_DIGITS)
    # Salva il DataFrame in un file Excel
    df_readout_MNIST.to_excel('Readout_on_MNIST_retrain_on_'+retrain_ds_type+'_'+type_retrain+'_'+type_mix+'.xlsx', index=False, header=False)
    Readout_last_layer_MNIST = df_readout_MNIST.values[:, len(dbn.rbm_layers)]

    return Readout_last_layer_MNIST, Readout_last_layer_RETRAINING_DS, dbn


def get_prototypes(Train_dataset,nr_categories=26):
  Prototypes = torch.zeros(nr_categories,Train_dataset['data'].shape[2])
  beg_range=0
  if nr_categories == 26:
    beg_range=10
  for l_idx in range(beg_range,beg_range+nr_categories):
    indices = torch.nonzero(Train_dataset['labels']==l_idx)

    sel_imgs = torch.zeros(indices.shape[0],Train_dataset['data'].shape[2])

    for c,idx in enumerate(indices):
      nB = int(idx[0])
      nwB = int(idx[1])
      sel_imgs[c,:] = Train_dataset['data'][nB,nwB,:]

    Avg_cat = torch.mean(sel_imgs,axis = 0)
    Prototypes[l_idx-10,:] = Avg_cat

  return Prototypes

'''
analysis prototypes:
MNIST_prototypes = get_prototypes(train_dataset,nr_categories=10)
EMNIST_prototypes = get_prototypes(train_dataset_retraining_ds,nr_categories=26)

Euclidean_dist_MNIST_EMNIST = torch.zeros(MNIST_prototypes.shape[0],EMNIST_prototypes.shape[0])

for i_MNIST, MNIST_prot in enumerate(MNIST_prototypes):
  for i_EMNIST, EMNIST_prot in enumerate(EMNIST_prototypes):
    Euclidean_dist_MNIST_EMNIST[i_MNIST, i_EMNIST] = torch.norm(MNIST_prot - EMNIST_prot)

Mins, _ =torch.min(Euclidean_dist_MNIST_EMNIST,axis=0)
topk_values, topk_indices = torch.topk(Mins, k=10)
alphabet = "abcdefghijklmnopqrstuvwxyz"

# Map indices to letters
letters = [alphabet[i] for i in topk_indices]

# Join the letters to form a string
result = ''.join(letters)

print("Letters with the specified indices:", result)

#vedere i prototipi
Letter_prototypes_28x28 = torch.zeros(26,28,28)
c=0
for lp in EMNIST_prototypes:
  image = lp.view(28, 28)
  # 1. Rotate the tensor 90 degrees in the opposite direction
  #image = torch.rot90(image, k=-1)

  # 2. Flip the tensor horizontally to restore it to its original orientation
  #image = torch.flip(image, [1])
  #Letter_prototypes_28x28[c,:,:] = image

  # Display the image using Matplotlib
  plt.imshow(image.cpu(), cmap='gray')
  plt.show()
  c=c+1


'''
