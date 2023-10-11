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

  upper = int(Xtrain.shape[0]*(4/5))
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
      upper = int(_Xtrain.shape[0]*(4/5))
      
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


def get_retraining_data(MNIST_train_dataset):
  #NOTA: il labelling dell'EMNIST by class ha 62 labels: le cifre (0-9), lettere MAUSCOLE (10-36), lettere MINUSCOLE(38-62)
  #20,000 uppercase letters from the first 10 EMNIST classes.
  nr_batches_retraining = round(20000/128)
  def decrease_labels_by_10(data): 
    image, label = data
    label -= 10  # Sottrai 10 da ciascuna etichetta
    return image, label
  #load  EMNIST byclass data
  transform =transforms.Compose([transforms.ToTensor()])
  data_train_EMNIST = datasets.EMNIST('../data', train=True,split = 'byclass', download=True, transform=transform)
  data_test_EMNIST = datasets.EMNIST('../data', train=False,split = 'byclass', download=True, transform=transform)
  target_classes = list(range(10, 20)) #i.e. the first 10 capital letter classes
  data_train_EMNIST = [item for item in data_train_EMNIST if item[1] in target_classes]
  data_test_EMNIST = [item for item in data_test_EMNIST if item[1] in target_classes]
  #i relabel data from 10-19 to 0-9
  data_train_EMNIST = [decrease_labels_by_10(item) for item in data_train_EMNIST]
  data_test_EMNIST = [decrease_labels_by_10(item) for item in data_test_EMNIST]

  train_data_EMNIST, train_labels_EMNIST = data_and_labels(data_train_EMNIST, BATCH_SIZE=128,NUM_FEAT=np.int32(28*28),DATASET_ID='MNIST',n_cols_labels=1)
  test_data_EMNIST, test_labels_EMNIST = data_and_labels(data_test_EMNIST, BATCH_SIZE=128,NUM_FEAT=np.int32(28*28),DATASET_ID='MNIST',n_cols_labels=1)
  
  #i select just 20000 examples
  train_data_EMNIST = train_data_EMNIST[:nr_batches_retraining,:,:]
  train_labels_EMNIST = train_labels_EMNIST[:nr_batches_retraining,:,:]
  train_dataset_EMNIST = {'data': train_data_EMNIST, 'labels': train_labels_EMNIST}
  test_dataset_EMNIST = {'data': test_data_EMNIST, 'labels': test_labels_EMNIST}

  half_EMNIST = train_dataset_EMNIST['data'][:(nr_batches_retraining//2),:,:].to('cuda')
  half_MNIST = MNIST_train_dataset['data'][:(nr_batches_retraining//2),:,:].to('cuda')
  mix_EMNIST_MNIST = torch.cat((half_MNIST, half_EMNIST), dim=0)
 
  # Generate a random permutation of indices
  permuted_indices = torch.randperm(nr_batches_retraining)
  # Use the permutation to shuffle the examples of the dataset
  mix_EMNIST_MNIST = mix_EMNIST_MNIST[permuted_indices]
  
  return train_dataset_EMNIST,test_dataset_EMNIST,mix_EMNIST_MNIST

def get_ridge_classifiers(MNIST_Train_DS, MNIST_Test_DS):
  Zambra_folder_drive = '/content/gdrive/My Drive/ZAMBRA_DBN/'
  MNIST_rc_file= os.path.join(Zambra_folder_drive,'MNIST_ridge_classifiers.pkl')
  EMNIST_rc_file= os.path.join(Zambra_folder_drive,'EMNIST_ridge_classifiers.pkl')
  print("\033[1m Make sure that your iDBN was trained only with MNIST for 100 epochs \033[0m")
  DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  dbn,train_dataset, test_dataset,classifier= tool_loader_ZAMBRA(DEVICE,  selected_idx = [20,31], half_data=False, only_data = False)
  if not(os.path.exists(MNIST_rc_file)):
    readout_acc_V, MNIST_classifier_list = readout_V_to_Hlast(dbn,MNIST_Train_DS, MNIST_Test_DS)
    # Save the list of classifiers to a file
    with open(MNIST_rc_file, 'wb') as file:
        pickle.dump(MNIST_classifier_list, file)
  else:
    with open(MNIST_rc_file, 'rb') as file:
      MNIST_classifier_list = pickle.load(file)# Load the list of classifiers from the file

  if not(os.path.exists(EMNIST_rc_file)):
    train_dataset_EMNIST,test_dataset_EMNIST,mix_EMNIST_MNIST =get_retraining_data(train_dataset)
    Xtrain = train_dataset_EMNIST['data'].to(DEVICE)
    Xtest  = test_dataset_EMNIST['data'].to(DEVICE)
    Ytrain = train_dataset_EMNIST['labels'].to(DEVICE)
    Ytest  = test_dataset_EMNIST['labels'].to(DEVICE)
    DATASET_ID='MNIST'
    with open(os.path.join(Zambra_folder_drive, f'lparams-{DATASET_ID.lower()}.json'), 'r') as filestream:
        LPARAMS = json.load(filestream)
    with open(os.path.join(Zambra_folder_drive, 'cparams.json'), 'r') as filestream:
        CPARAMS = json.load(filestream)
    LPARAMS['EPOCHS']=100
    READOUT = CPARAMS['READOUT']
    NUM_DISCR = CPARAMS['NUM_DISCR']
    dbn.train(Xtrain, Xtest, Ytrain, Ytest, LPARAMS, readout = READOUT, num_discr = NUM_DISCR)
    readout_acc_V, EMNIST_classifier_list = readout_V_to_Hlast(dbn,train_dataset_EMNIST,test_dataset_EMNIST)
    # Save the list of classifiers to a file
    with open(EMNIST_rc_file, 'wb') as file:
        pickle.dump(EMNIST_classifier_list, file)
  else:
    with open(EMNIST_rc_file, 'rb') as file:
      EMNIST_classifier_list = pickle.load(file)# Load the list of classifiers from the file
  return MNIST_classifier_list, EMNIST_classifier_list