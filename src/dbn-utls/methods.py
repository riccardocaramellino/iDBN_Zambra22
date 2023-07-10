import torch
import math
import copy
import random
import matplotlib.pyplot as plt
import numpy as np


def label_biasing(model, on_digits=1, topk = 149):

        # aim of this function is to implement the label biasing procedure described in
        # https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00515/full
        

 
        # Now i set the label vector from which i will obtain the hidden layer of interest 
        Biasing_vec = torch.zeros (model.Num_classes,1, device = model.DEVICE)
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

def generate_from_hidden(model, input_hid_prob , nr_gen_steps, temperature=1, include_energy = 0):

    img_side = int(np.sqrt(model.vishid.shape[0]))
    #input_hid_prob has size Nr_hidden_units x num_cases. Therefore i transpose it
    input_hid_prob = torch.transpose(input_hid_prob,0,1)

    numcases = input_hid_prob.size()[0] #numbers of samples to generate

    hidden_layer_size = input_hid_prob.size()[1]

    hid_prob = torch.zeros(numcases,hidden_layer_size, nr_gen_steps, device=model.DEVICE)
    hid_states = torch.zeros(numcases,hidden_layer_size, nr_gen_steps, device=model.DEVICE)

    vis_prob = torch.zeros(numcases, img_side*img_side, nr_gen_steps, device=model.DEVICE)
    vis_states = torch.zeros(numcases ,img_side*img_side, nr_gen_steps, device=model.DEVICE)

    Energy_matrix = torch.zeros(numcases, nr_gen_steps, device=model.DEVICE)


    for step in range(0,nr_gen_steps):
        
        if step>0: #if we are after the 1st gen step...
            
            hid_activation = torch.matmul(vis_states[:,:,step-1], model.vishid) + model.hidbiases # hid_act = V(s-1)*W + bH
            
            # Pass the activation in a sigmoid
            if temperature==1:
                hid_prob[:,:,step]  = torch.sigmoid(hid_activation)
            elif isinstance(temperature, list):
                hid_prob[:,:,step]  = torch.sigmoid(hid_activation/temperature[step])
            else:
                hid_prob[:,:,step]  = torch.sigmoid(hid_activation/temperature)

            if model.Hidden_mode=='binary': #If the hidden layer is set to be binary...
                hid_states[:,:,step] = torch.bernoulli(hid_prob[:,:,step]) #do the bernoullian sampling
            else:
                hid_states[:,:,step] = hid_prob[:,:,step] #if the hidden layer is set to be continous, avoid the bernoullian sampling

        else: #if it is the 1st step of generation...
            hid_prob[:,:,step]  = input_hid_prob #the hidden probability is the one in the input
            hid_states[:,:,step]  = input_hid_prob #the hidden probability is the one in the input


        vis_activation = torch.matmul(hid_states[:,:,step],torch.transpose(model.vishid, 0, 1)) + model.visbiases #V(s) = H(s)*W + bV
        
        #pass the visible activation through the sigmoid to obtain the visible probabilities
        if temperature==1:
            vis_prob[:,:,step]  = torch.sigmoid(vis_activation)
        elif isinstance(temperature, list):
            vis_prob[:,:,step]  = torch.sigmoid(vis_activation/temperature[step])
        else:
            vis_prob[:,:,step]  = torch.sigmoid(vis_activation/temperature)


        if model.Visible_mode=='binary': # do the bernoullian sampling if Visible_mode is set to be binary
            vis_states[:,:,step] = torch.bernoulli(vis_prob[:,:,step])
        elif model.Visible_mode=='continous':
            vis_states[:,:,step] = vis_prob[:,:,step]

        if  include_energy == 1: #if you want also to compute energy...
            state_energy = model.energy_f(hid_states[:,:,step], vis_states[:,:,step])
            Energy_matrix[:,step] = state_energy[:,0]


    result_dict = dict(); 
    result_dict['hid_states'] = hid_states
    result_dict['vis_states'] = vis_states
    result_dict['Energy_matrix'] = Energy_matrix
    result_dict['hid_prob'] = hid_prob
    result_dict['vis_prob'] = vis_prob


    return result_dict

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



