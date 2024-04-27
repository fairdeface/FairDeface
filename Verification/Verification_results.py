from deepface import DeepFace
from shutil import rmtree
import pickle
import numpy as np
from PIL import Image
from os.path import join, isdir, isfile, basename
from os import listdir, remove, walk, makedirs
import os
import json, pickle
import requests
from random import shuffle, choice
import argparse

parser = argparse.ArgumentParser()# Add an argument
parser.add_argument('--file_path', type=str, required=True)# Parse the argument
parser.add_argument('--fig_style', type=str,  default='all',  help=' could be <all> to show the results for all the models in a single plot')# Parse the argument

args = parser.parse_args()
verification_all_file = args.file_path

fig_style = args.fig_style
# fig_style='all'
# verification_all_file='Verification_DP1_same-photo_003433.pkl'
if 'same-photo' in verification_all_file:
    ch = 'same-photo'
    ylabel = 'Obfuscation Success Rates'
elif 'different-photo' in verification_all_file:
    ch = 'different-photo'
    ylabel = 'Obfuscation Success Rates'
elif 'same-identity' in  verification_all_file:
    ch = 'same-identity'
    ylabel = 'True Positive Rates'
elif 'different-identity' in verification_all_file:
    ch = 'different-identity'
    ylabel = 'True Negative Rates'

def decission_maker(distance, threshold):
    if ch=='same-photo' or ch =='different-photo' or ch=='different-identity':
        return distance>threshold
    elif ch=='same-identity':
        return distance<threshold
    

distances_demo ={}
# def check_if_the_file_exist(verification_all_file):
#     i = 4
#     pickles = listdir('.')
#     pickles = [p for p in pickles if ".pkl" in p and verification_all_file[:-i] in p ]
   
#     last_ind = 0
#     for p in pickles:
#         pi = int( p[len(verification_all_file[:-i]):-i])
#         if pi>last_ind:
#            last_ind =pi
#            last = p
#     print("last_ind =", last_ind)
#     flag = 1
#     while(flag):
#         try:
#             with open(last, 'rb') as my_file:
#                 verification_all = pickle.load(my_file)
#                 my_file.close()
#                 flag = 0
#                 print("loaded_last_ind =", last_ind)
#                 for k in range(1, last_ind):
#                     file = verification_all_file[:-i] + format(k, '06d') + verification_all_file[-i:]
#                     if isfile(file):
#                         remove(file)
#                 return verification_all, last_ind+1

#         except:
#                remove(last)
#                last = verification_all_file[:-i] + format(last_ind-1, '06d') + verification_all_file[-i:]
#                last_ind = last_ind-1
    
# verification_all, last_ind = check_if_the_file_exist(verification_all_file)
with open(verification_all_file, 'rb') as my_file:
    verification_all = pickle.load(my_file)
    my_file.close()
thresholds={}
method= list(verification_all.keys())[0]
for model in verification_all[method].keys():
    distances_demo[model] = {}
    for dataset in verification_all[method][model].keys():
        distances_demo[model][dataset] = {}
        for demo in verification_all[method][model][dataset].keys():
            distances_demo[model][dataset][demo] = {}
            distances_demo[model][dataset][demo]['verified']= []
            distances_demo[model][dataset][demo]['distance']= []
            for person in verification_all[method][model][dataset][demo].keys():
                if type(verification_all[method][model][dataset][demo][person]) is dict:
                    for ver_pic_results in verification_all[method][model][dataset][demo][person]['distance']:
                        distances_demo[model][dataset][demo]['verified'].append(ver_pic_results['verified'])
                        distances_demo[model][dataset][demo]['distance'].append(float(ver_pic_results['distance']))
    
    thresholds[model]= int(100*ver_pic_results['threshold'])
thresholds['ArcFace']=75
datasets=['BFW', 'DemogPairs']
pair_datasets = [d for d in distances_demo[model].keys() if d in datasets ]
#success rates for thresholds 0 to 100                        
success_rates={}                    
total_pairs={}
for model in distances_demo.keys():
    success_rates[model] = {}
    total_pairs[model]={}

    for dataset in pair_datasets:
        total_pairs[model][dataset]={}

        success_rates[model][dataset] = {}
        for demo in distances_demo[model][dataset].keys():
            total_pairs[model][dataset][demo]=len(distances_demo[model][dataset][demo]['distance'])

            success_rates[model][dataset][demo] = {}   
            for d in range(101):
                summ = 0
                for distance in distances_demo[model][dataset][demo]['distance']:
                        summ +=decission_maker(100*distance, d)
                try:
                    success_rates[model][dataset][demo][d] = ( summ/len(distances_demo[model][dataset][demo]['distance']))
                except Exception as e:
                    print(e)


#Races
Races= {"BFW": ['white', 'indian', 'black', 'asian'], "DemogPairs": ['White', 'Black', 'Asian'], "RFW": ['Asian',  'African', 'Caucasian', 'Indian']}
# Races= {"BFW": ['white', 'indian', 'black', 'asian'], "DemogPairs": ['White', 'Black', 'Asian']}
race_datasets= [d for d in verification_all[method][model].keys() if d in Races.keys()]

success_rates_race = {}
total_race = {}

for model in distances_demo.keys():
    total_race[model]={}
    sum_race = {}
    for dataset in race_datasets:
        total_race[model][dataset] = {}
        sum_race[dataset]= {}
        for race in Races[dataset]:
            total_race[model][dataset][race] = 0
            sum_race[dataset][race]= {}
            for d in range(101):
                sum_race[dataset][race][d]= 0
    
    success_rates_race[model] = {}
    for dataset in race_datasets:
        success_rates_race[model][dataset] = {}
        for demo in distances_demo[model][dataset].keys(): 
            for race in Races[dataset]:
                success_rates_race[model][dataset][race] = {}
                if race in demo:
                    race_p = race
                    total_race[model][dataset][race_p] += len(distances_demo[model][dataset][demo]['distance'])
            for d in range(101):
                for distance in distances_demo[model][dataset][demo]['distance']:
                        sum_race[dataset][race_p][d] += decission_maker(100*distance, d)

        for race in Races[dataset]:
           for d in range(101):
               try:
                   success_rates_race[model][dataset][race][d] = sum_race[dataset][race][d]/total_race[model][dataset][race]
               except Exception as e:
                    print(e)
           
 
#Genders
datasets= {"BFW", "DemogPairs", "CelebA"}
Genders = ["Males", "Females"]

gen_datasets= [d for d in verification_all[method][model].keys() if d in datasets]
success_rates_Gender = {}
total_Gender = {}

for model in distances_demo.keys():
    
    sum_Gender = {}
    total_Gender[model] = {}
    for dataset in gen_datasets:
        total_Gender[model][dataset] = {}
        sum_Gender[dataset]= {}
        for Gender in Genders:
            total_Gender[model][dataset][Gender] = 0
            sum_Gender[dataset][Gender]= {}
            for d in range(101):
                sum_Gender[dataset][Gender][d]= 0
    
    success_rates_Gender[model] = {}
    for dataset in gen_datasets:
        success_rates_Gender[model][dataset] = {}
        for Gender in Genders:
            success_rates_Gender[model][dataset][Gender] = {}
            
        for demo in distances_demo[model][dataset].keys(): 
                
            if 'female' in demo or 'Female' in demo:
                Gender = "Females"
            else:
                Gender = "Males"
            total_Gender[model][dataset][Gender] += len(distances_demo[model][dataset][demo]['distance'])
            for d in range(101):
                for distance in distances_demo[model][dataset][demo]['distance']:
                        sum_Gender[dataset][Gender][d] += decission_maker(100*distance, d)

        for Gender in Genders:
            for d in range(101):
                try:
                    success_rates_Gender[model][dataset][Gender][d] = sum_Gender[dataset][Gender][d]/total_Gender[model][dataset][Gender]
                except Exception as e:
                    print(e)



#success rates based on the models suggestions    
success_rates_verified={}                    
for model in distances_demo.keys():
    success_rates_verified[model] = {}
    for dataset in distances_demo[model].keys():
        success_rates_verified[model][dataset] = {}
        for demo in distances_demo[model][dataset].keys():  
            summ = 0
            for verified in distances_demo[model][dataset][demo]['verified']:
                if verified == False:
                    summ+=1
            try:
                success_rates_verified[model][dataset][demo] = summ/len(distances_demo[model][dataset][demo]['verified'])
            except Exception as e:
                print(e)





# #Races Verified
# # Races= {"BFW": ['white', 'indian', 'black', 'asian'], "DemogPairs": ['White', 'Black', 'Asian'], "RFW": ['Asian',  'African', 'Caucasian', 'Indian']}
# Races= {"BFW": ['white', 'indian', 'black', 'asian'], "DemogPairs": ['White', 'Black', 'Asian']}

# success_rates_race_verified = {}
# for model in distances_demo.keys():
    
#     sum_race = {}
#     total_race = {}
#     for dataset in race_datasets:
#         total_race[dataset] = {}
#         sum_race[dataset]= {}
#         for race in Races[dataset]:
#             total_race[dataset][race] = 0
#             sum_race[dataset][race]= 0
            
#     success_rates_race_verified[model] = {}
#     for dataset in race_datasets:
#         success_rates_race_verified[model][dataset] = {}
#         for demo in distances_demo[model][dataset].keys(): 
#             for race in Races[dataset]:
#                 success_rates_race_verified[model][dataset][race] = {}
#                 if race in demo:
#                     race_p = race
#                     total_race[dataset][race_p] += len(distances_demo[model][dataset][demo]['distance'])

#             for verified in distances_demo[model][dataset][demo]['verified']:
#                 if verified == False:
#                     sum_race[dataset][race_p]+=1

#         for race in Races[dataset]:
#             try:
#               success_rates_race_verified[model][dataset][race] = sum_race[dataset][race]/total_race[dataset][race]
#             except Exception as e:
#                 print(e)
              

# success_rates_Gender_verified = {}
# for model in distances_demo.keys():
    
#     sum_Gender = {}
#     total_Gender = {}
#     for dataset in gen_datasets:
#         total_Gender[dataset] = {}
#         sum_Gender[dataset]= {}
#         for Gender in Genders:
#             total_Gender[dataset][Gender] = 0
#             sum_Gender[dataset][Gender]= 0

    
#     success_rates_Gender_verified[model] = {}
#     for dataset in gen_datasets:
#         success_rates_Gender_verified[model][dataset] = {}
#         for Gender in Genders:
#             success_rates_Gender_verified[model][dataset][Gender] = {}
            
#         for demo in distances_demo[model][dataset].keys(): 
                
#             if 'female' in demo or 'Female' in demo:
#                 Gender = "Females"
#             else:
#                 Gender = "Males"
#             total_Gender[dataset][Gender] += len(distances_demo[model][dataset][demo]['distance'])
        
#             for verified in distances_demo[model][dataset][demo]['verified']:
#                 if verified == False:
#                     sum_Gender[dataset][Gender]+=1

#         for Gender in Genders:
#             try:
#                 success_rates_Gender_verified[model][dataset][Gender] = sum_Gender[dataset][Gender]/total_Gender[dataset][Gender]                 
#             except Exception as e:
#                 print(e)
               

success_rates_all= {}
success_rates_all['success_rates'] = success_rates
success_rates_all['success_rates_race'] = success_rates_race
success_rates_all['success_rates_Gender'] = success_rates_Gender

# success_rates_all['success_rates_verified'] = success_rates_verified
# success_rates_all['success_rates_race_verified'] = success_rates_race_verified
# success_rates_all['success_rates_Gender_verified'] = success_rates_Gender_verified


###############################################################################################
# with open('SuccessRates_' + method + "_" + ch +'.pkl', 'wb') as file:
#     pickle.dump(success_rates_all, file)
import matplotlib    
import pandas as pd
import pickle
from os import listdir
import pickle, json
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 7})
fontsize = 15

markers = ["O", "v", "1", "s", "p", "X", "|", "$H$" ]
color_line='magenta'
datasets = ['BFW', 'DemogPairs']
colors = ['red', 'blue', 'yellow',  'green', 'purple', 'black', 'pink', 'gold' ]
color_dic = {"white_males": colors[0], "White_Males": colors[0], "black_males": colors[1], "Black_Males": colors[1],  
              "asian_males": colors[2], "Asian_Males": colors[2], "indian_males": colors[3], "Indian_Males": colors[3], 
             
              "white_females": colors[4], "White_Females": colors[4], "indian_females": colors[5], "Indian_Females": colors[5],  
              "black_females": colors[6], "Black_Females": colors[6], "asian_females": colors[7], "Asian_Females": colors[7], 
              "white": colors[0], "White": colors[0], "Caucasian": colors[0], 'indian': colors[3], 'Indian': colors[3],
              'black': colors[1], 'Black': colors[1], 'African': colors[1], 'asian': colors[2], 'Asian': colors[2], 
              'males': colors[0], 'Males': colors[0], "females": colors[1], "Females": colors[1]} 

markers = ["o", "v", "1", "s", "p", "X", "|", "$H$" ]
marker_dic = {"white_males": markers[0], "White_Males": markers[0], "black_males": markers[1], "Black_Males": markers[1],  
              "asian_males": markers[2], "Asian_Males": markers[2], "indian_males": markers[3], "Indian_Males": markers[3], 
             
              "white_females": markers[4], "White_Females": markers[4], "indian_females": markers[5], "Indian_Females": markers[5],  
              "black_females": markers[6], "Black_Females": markers[6], "asian_females": markers[7], "Asian_Females": markers[7], 
              "white": markers[0], "White": markers[0], "Caucasian": markers[0], 'indian': markers[3], 'Indian': markers[3],
              'black': markers[1], 'Black': markers[1], 'African': markers[1], 'asian': markers[2], 'Asian': markers[2], 
              'males': markers[0], 'Males': markers[0], "females": markers[1], "Females": markers[1]} 


def plot_main(rates, term):

    
    SR_means = {}
    for model in rates.keys():
        SR_means[model]={}
        for dataset in rates[model].keys():
            SR_means[model][dataset]={}
            for demo in rates[model][dataset].keys():
                SR_means[model][dataset][demo] = np.mean(list(rates[model][dataset][demo].values()))
    
    datasets = list(rates[model].keys())

    len_datasets = len(datasets)
    len_models=1

    if fig_style=='all':
        row = 0
        len_models= len(list(rates.keys()))
        fig2, ax = plt.subplots( len_models,len_datasets, sharex='col', sharey='row', figsize=(  5*len_datasets, 5*len_models))  # Opening pickle file
        for model in rates.keys():         
            col = 0
       
            try:
                if row==len_models-1:
                    last=1
                else:
                    last=0
                for dataset in datasets:
                    dataset_rates = rates[model][dataset]
                    if len_models==1:
                        sub_plot(ax[col], dataset_rates, ch, dataset,last, model)
                        ax[col].axvline(x = thresholds[model], color = color_line)
            
                    else:
                        sub_plot(ax[row,col], dataset_rates, ch, dataset,last, model)
                        ax[row,col].axvline(x = thresholds[model], color = color_line)
                    col+=1
                 
                row += 1
       
            except Exception as e:
                print(model, " has problems ")
                print(e)
                print(row)
                print(col)
        # manager = plt.get_current_fig_manager()
        # manager.full_screen_toggle()
        # plt.tight_layout()
        
        directory = join('Verification_Rates', 'All')
        
            
        if not isdir(directory):
            makedirs(directory)
        figname="Verification-" + method + "-" + ch + "-" + term + "-all_models.jpg"
        figname2 = join(directory, figname)
        plt.gcf().set_size_inches(4.5*len_datasets, 3*len_models)
        plt.tight_layout()

        # fig2.subplots_adjust(wspace=0.3, hspace=0.3) 
        fig2.savefig(figname2, dpi=150)

    else:
        len_models == 1
        for model in rates.keys(): 
            fig2, ax = plt.subplots( len_models,len_datasets, sharex='col', sharey='row', figsize=(5*len_datasets, 5*len_models))  # Opening pickle file
        
            # fig2.tight_layout()
            col = 0
    
            try:
               
                for dataset in datasets:
                    dataset_rates = rates[model][dataset]
                    if len_models==1:
                        last = 1
                        sub_plot(ax[col], dataset_rates, ch, dataset,last, model)
                        ax[col].axvline(x = thresholds[model], color = color_line)
                        col+=1
            
                    # else:
                    #     sub_plot(ax[row,col], dataset_rates, ch, dataset,last, model)
                    #     ax[row,col].axvline(x = thresholds[model], color = color_line)
                    # col+=1
                 
                # row += 1
    
            except Exception as e:
                print(model, " has problems ")
                print(e)
                # print(row)
                print(col)
            # manager = plt.get_current_fig_manager()
            # manager.full_screen_toggle()
            # plt.tight_layout()
            directory = join('Verification_Rates', 'Separate', method)
            
                
            if not isdir(directory):
                makedirs(directory)
            figname="Verification-" + method + "-" + ch + "-" + term + '-' + model+ ".jpg"
            figname2 = join(directory, figname)
            plt.gcf().set_size_inches(4.5*len_datasets, 3*len_models)
            plt.tight_layout()

            fig2.savefig(figname2, dpi=150)




def sort_legend_mean_based(plt, dataset):

    demo_means= []
    for demo in dataset.keys():
        demo_means.append(np.mean(list(dataset[demo].values())))
    # demo_means = np.mean(list(dataset.values()), axis=1)

    sorted_m = sorted(range(len(demo_means)), key=lambda x:demo_means[x], reverse=True)

    # reordering the labels
    handles, labels = plt.get_legend_handles_labels()
    # specify order
    order = sorted_m

    # pass handle & labels lists along with order as below
    plt.legend([handles[i] for i in order], [labels[i] for i in order])
    return plt

def sort_legend(plt, dataset, threshold):
    # demo_arcs= []
    # for demo in dataset.keys():
    #     demo_arcs.append(np.mean(list(dataset[demo].values())))
    demo_arcs =[]
    for demo in dataset.keys():
        demo_arcs.append(dataset[demo][threshold])
    sorted_m = sorted(range(len(demo_arcs)), key=lambda x:demo_arcs[x], reverse=True)

    # reordering the labels
    handles, labels = plt.get_legend_handles_labels()
    # specify order
    order = sorted_m

    # pass handle & labels lists along with order as below
    plt.legend([handles[i] for i in order], [labels[i] for i in order])
    return plt

def sub_plot(ax, dataset, ch, dataset_str, last, model):
    print(dataset_str)
    x_axis= list(range(101))
    for demo in dataset.keys():
        ax.plot(x_axis, list(dataset[demo].values()),  color=color_dic[demo], marker=marker_dic[demo], label=demo, markersize=2)
    # if method=='Org':
    #     ylabel="True Positive Rate"
    # else:
    #     ylabel= "Obfuscation Success Rate"
    ax.set_ylabel(ylabel)
    if last:
        ax.set_xlabel("Distance")

    ax.set_title(method + "-" + model + "-" + dataset_str + " - " + ch)
    ax = sort_legend(ax, dataset, threshold=thresholds[model]) 


models = list(success_rates.keys())
len_models= len(models)

# for model in success_rates_all['success_rates'].keys():    
    
plot_main(success_rates, 'pairs')

        
# #################################################################################
#Race
plot_main(success_rates_race, 'race')

# #################################################################################
#Gender
plot_main(success_rates_Gender, 'gender')

####################################################################################
# plt.close('all')

###demographic parities 
fairness_thresh=.8
import pickle
import seaborn as sns # for data visualization
import pandas as pd # for data analysis
import numpy as np # for numeric calculation
from os.path import join, isdir, isfile, basename
from os import listdir, remove, walk, mkdir
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt2 # for data visualization

# my_colors = sns.color_palette("viridis",5)
# my_cmap = ListedColormap(my_colors)
# my_colors=['red', 'orange', 'green', 'yellow', 'blue']
my_colors=['red', 'green']

my_cmap= ListedColormap(my_colors)
bounds = [0, .8,  1]
my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))
patterns = ['', 'oo', '////', 'XXX', '*']

# sns.set(font_scale=.8)
def sort_demos(total_demos, threshold):
    new_demos= {}
    for demo in total_demos.keys():
        if type(total_demos[demo])==dict:
            new_demos[demo]= total_demos[demo][threshold]
    new_demos = sorted(new_demos.items(), key=lambda x:x[1],  reverse=True)
    new_demos = [d[0] for d in new_demos]
    return new_demos
def plot_demographic_parity(rates, d_type):

    if fig_style=='all':
        models= list(rates.keys())
        num_rows =len(models)
        datasets = list(rates[models[0]].keys())
        num_cols= len(datasets)
        fig, ax = plt2.subplots(num_rows, num_cols, figsize=( 5*num_rows, 5*num_cols))
        figname = "Bias-Verification-" + method + "-" + ch+ '-' + d_type + '-' + 'all_models' 
        d=1
        for model in rates.keys():
            biases={}
            
            # datasets = list(rates[model].keys())
            # num_cols= len(datasets)
            # num_rows= int(np.ceil(len(datasets)/2))
            for dataset in datasets:
                
                biases[dataset]={}
                
                demos = list(rates[model][dataset].keys())
        
                num_demos =len(demos)
                demos=sort_demos(rates[model][dataset], thresholds[model])
                dataset_biass = np.empty((num_demos, num_demos))
                Demos=[]
                for D in demos:
                    if "_" in D:
                        D = ''.join([x[0].upper() for x in D.split('_')])
                    Demos.append(D)
        
                i=0
        
                for demo in demos:
                    biases[dataset][demo]={}
                    for demo2 in demos:
                        try:
                            biases[dataset][demo][demo2]= (rates[model][dataset][demo][thresholds[model]])/(rates[model][dataset][demo2][thresholds[model]])
                            if biases[dataset][demo][demo2]<=fairness_thresh:
                                print( model, dataset, demo, demo2, biases[dataset][demo][demo2])
                        except:
                                biases[dataset][demo][demo2]= float("NaN")
                                print('devided by zero',model, dataset, demo2) 
                                
                    dataset_biass[i,:]= np.array(list(biases[dataset][demo].values()))
                    i+=1
                 # x = np.array(biases[att][method][dataset].values())
                dataset_biass = np.round(dataset_biass, 2)
                 # corr= dataset_biass.corr()
                
                 # Getting the Upper Triangle of the co-relation matrix
                matrix = np.triu(dataset_biass)
                annot_kws={
                 'size':100,
                'fontstyle':'italic',  
                'color':"k",
                'rotation':"vertical",
                'verticalalignment':'center',
                'backgroundcolor':'w'}
                plt2.subplot(num_rows, num_cols, d)
                plt2.title(model + '-' + dataset)
        
                    
                d+=1
                sns.heatmap(dataset_biass,  xticklabels=Demos, yticklabels=Demos,  mask= matrix,  cmap=my_cmap,  norm=my_norm) 
        # plt.show()
        directory = join('Verification_Rates', 'All')
        
            
        if not isdir(directory):
            makedirs(directory)
        figname2 = join(directory, figname)
        plt2.gcf().set_size_inches( 3*num_cols, 3*num_rows)
        plt2.tight_layout()

        # plt2.subplots_adjust(wspace=0.7, hspace=0.5) 

        plt2.savefig(figname2+'.jpg', dpi=150)
                    
    else:
            
        for model in rates.keys():
            figname = "Bias-Verification-" + method + "-" + ch+ '-' + d_type + '-' + model 
            biases={}
            d=1
            datasets = list(rates[model].keys())
            num_rows = 1
            # num_rows= int(np.ceil(len(datasets)/2))
            num_cols=len(datasets)
            fig, ax = plt2.subplots(num_rows, num_cols, figsize=(  5*num_rows, 5*num_cols))

            for dataset in datasets:

                biases[dataset]={}
                
                demos = list(rates[model][dataset].keys())
        
                num_demos =len(demos)
                demos=sort_demos(rates[model][dataset], thresholds[model])

                dataset_biass = np.empty((num_demos, num_demos))
                Demos=[]
                for D in demos:
                    if "_" in D:
                        D = ''.join([x[0].upper() for x in D.split('_')])
                    Demos.append(D)
        
                i=0
        
                for demo in demos:
                    biases[dataset][demo]={}
                    for demo2 in demos:
                        try:
                            biases[dataset][demo][demo2]= (rates[model][dataset][demo][thresholds[model]])/(rates[model][dataset][demo2][thresholds[model]])
                            if biases[dataset][demo][demo2]<=fairness_thresh:
                                print( model, dataset, demo, demo2, biases[dataset][demo][demo2])
                        except:
                                biases[dataset][demo][demo2]= float("NaN")
                                print('devided by zero',model, dataset, demo2) 
                                
                    dataset_biass[i,:]= np.array(list(biases[dataset][demo].values()))
                    i+=1
                 # x = np.array(biases[att][method][dataset].values())
                dataset_biass = np.round(dataset_biass, 2)
                 # corr= dataset_biass.corr()
                
                 # Getting the Upper Triangle of the co-relation matrix
                matrix = np.triu(dataset_biass)
                annot_kws={
                 'size':100,
                'fontstyle':'italic',  
                'color':"k",
                'rotation':"vertical",
                'verticalalignment':'center',
                'backgroundcolor':'w'}
                plt2.subplot(num_rows,num_cols, d)
                plt2.title(dataset)
        
                    
                d+=1
                sns.heatmap(dataset_biass,  xticklabels=Demos, yticklabels=Demos,  mask= matrix,  cmap=my_cmap,  norm=my_norm) 
            # plt.show()
            # plt2.tight_layout()
            directory = join('Verification_Rates', 'Separate', method)
            
                
            if not isdir(directory):
                makedirs(directory)
            figname2 = join(directory, figname)
            plt2.gcf().set_size_inches(3*num_cols, 3*num_rows)
            plt2.tight_layout()

            plt2.savefig(figname2+'.jpg', dpi=150)
                    

plot_demographic_parity(success_rates, 'pairs')
plot_demographic_parity(success_rates_race, 'race')
plot_demographic_parity(success_rates_Gender, 'gender')                
#####################################################################################
fontsize = 12

hatches = ['/', '\\', '//', '-', '+', 'x', 'o', 'O', '.', '*']
hatch_dic = {"white_males": hatches[0], "White_Males": hatches[0], "black_males": hatches[1], "Black_Males": hatches[1],
             "asian_males": hatches[2], "Asian_Males": hatches[2], "indian_males": hatches[3], "Indian_Males": hatches[3],

             "white_females": hatches[4], "White_Females": hatches[4], "indian_females": hatches[5], "Indian_Females": hatches[5],
             "black_females": hatches[6], "Black_Females": hatches[6], "asian_females": hatches[7], "Asian_Females": hatches[7],
             "white": hatches[0], "White": hatches[0], "Caucasian": hatches[0], 'indian': hatches[3], 'Indian': hatches[3],
             'black': hatches[1], 'Black': hatches[1], 'African': hatches[1], 'asian': hatches[2], 'Asian': hatches[2],
             'males': hatches[0], 'Males': hatches[0], "females": hatches[1], "Females": hatches[1]}
line_styles=['-', '--', '-.', ':','dotted'  ]

def mean_rate_calculator(rates, threshold, total_nums):
        sum_all=0
        sum_rates=0
        means_dataset={}
        for dataset in rates.keys():
            sum_dataset=0
            sum_rates_dataset=0
            for demo in rates[dataset].keys():
                if type(rates[dataset][demo])==dict:
                    
                    num_pairs= total_nums[dataset][demo]
                    score=rates[dataset][demo][threshold]*num_pairs

                        
                    sum_rates_dataset+=score
                    sum_dataset+=num_pairs
                    
                    sum_rates+=score
                    sum_all+=num_pairs

            means_dataset[dataset]= sum_rates_dataset/sum_dataset
        means_total=sum_rates/sum_all
        return means_total, means_dataset        

def plot_main(rates,fig_style, term, ch, total_nums):
    if fig_style == 'all':
        models = list(rates.keys())
        num_rows = len(models)
        num_cols = 1
        fig, ax = plt.subplots( num_rows, num_cols, figsize=(10*num_cols, 5*num_rows))

        figname="Verification-Bars-" + method + "-" + ch + "-" + term +  "-all_models.jpg"

        Total_length = .9
        d = 0
        for model in rates.keys():
            datasets = list(rates[models[0]].keys())
            d += 1
            plt.subplot(num_rows, num_cols, d)
            plt.title(method + '-' + model, fontsize=fontsize)

            J = 0
            # fig2 = plt.figure('rates Rates for ' + method + '-' + model )
            X_axis = np.arange(len(datasets))
            for dataset in datasets:
                demos = list(rates[model][dataset].keys())
                demos=sort_demos(rates[model][dataset], thresholds[model])
                num_demos = len(demos)
                width = Total_length/(len(demos))
                i = 1/2
                for demo in demos:
                    plt.bar(J - Total_length/2 + width*i,  rates[model][dataset][demo][thresholds[model]], width, hatch=hatch_dic[demo], label=demo, color=color_dic[demo])
                    i += 1

                J += 1

                plt.xticks(X_axis, datasets, fontsize=fontsize)
                plt.xlabel("Datasets", fontsize=fontsize)
                if method!='Original':
                    plt.ylabel( "Obfuscation Success Rates", fontsize=fontsize)
                else:
                    plt.ylabel( "Verification Rates", fontsize=fontsize)

                    
                # plt.title("Number of Students in each group")

                # , loc="upper left", ncol=2, bbox_to_anchor=(0.5, 1.0)
         

            means_total, mean_dataset= mean_rate_calculator(rates[model],thresholds[model], total_nums[model])
           
            plt.axhline(y = means_total, color = colors[0], linestyle = line_styles[0], label = "Total: "+"{:.2f}".format(means_total)) 
            cl=0
            for dataset in mean_dataset.keys():
                cl+=1
                plt.axhline(y = mean_dataset[dataset], color = colors[cl], linestyle = line_styles[cl], label = dataset+": {:.2f}".format(mean_dataset[dataset])) 
                # plt.text(-1,mean_dataset[dataset], "{:.0f}".format(mean_dataset[dataset]), color=colors[cl],   ha="left", va="center")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), fontsize=fontsize-2, loc='center left', bbox_to_anchor=(1, 0.5))
        directory = join('Verification_Rates', 'All')
    
        figname2 = join(directory, figname)
        plt.gcf().set_size_inches(8*num_cols, 4*num_rows)
        plt.tight_layout()
        fig.subplots_adjust(wspace=0.3, hspace=0.3) 
        # fig.subplots_adjust(wspace=0.6, hspace=0.3) 

        fig.savefig(figname2, dpi=150)
        plt.close('all')
    else:

        Total_length = .9
        for model in rates.keys():
             # models = list(rates.keys())
            num_rows = 1
            num_cols = 1
            fig, ax = plt.subplots( num_rows, num_cols, figsize=(10*num_cols, 5*num_rows))
            figname="Verification-Bars-" + method + "-" + ch + "-" + term + '-' + model+ ".jpg"
            J = 0
            datasets = list(rates[model].keys())
            # fig2 = plt.figure('rates Rates for ' + method + '-' + model )
            X_axis = np.arange(len(datasets))
            for dataset in datasets:
                demos = list(rates[model][dataset].keys())
                demos=sort_demos(rates[model][dataset], thresholds[model])

                num_demos = len(demos)
                width = Total_length/(len(demos))
                i = 1/2
                for demo in demos:
                    plt.bar(J - Total_length/2 + width*i, rates[model][dataset][demo][thresholds[model]], width, hatch=hatch_dic[demo], label=demo, color=color_dic[demo])
                    i += 1

                J += 1

                plt.xticks(X_axis, datasets, fontsize=fontsize)
                plt.xlabel("Datasets", fontsize=fontsize)
                if method!='Original':
                    plt.ylabel( "Obfuscation Success Rates", fontsize=fontsize)
                else:
                    plt.ylabel( "Verification Rates", fontsize=fontsize)
  

            means_total, mean_dataset= mean_rate_calculator(rates[model],thresholds[model], total_nums[model])
           
            plt.axhline(y = means_total, color = colors[0], linestyle = line_styles[0], label = "Total:"+"{:.2f}".format(means_total)) 
            cl=0
            for dataset in mean_dataset.keys():
                cl+=1
                plt.axhline(y = mean_dataset[dataset], color = colors[cl], linestyle = line_styles[cl], label = dataset+":{:.2f}".format(mean_dataset[dataset])) 
                # plt.text(-1,mean_dataset[dataset], "{:.0f}".format(mean_dataset[dataset]), color=colors[cl],   ha="left", va="center")
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), fontsize=fontsize-2, loc='center left', bbox_to_anchor=(1, 0.5))
            
            plt.tight_layout()
            plt.title(method + '-' + model, fontsize=fontsize)
            directory = join('Verification_Rates', 'Separate', method)
            plt.tight_layout()

            figname2 = join(directory, figname)
            plt.gcf().set_size_inches(10*num_cols, 4*num_rows)
            # fig.subplots_adjust(wspace=0.3, hspace=0.3) 

            fig.savefig(figname2, dpi=150)
            plt.close('all')


plot_main(success_rates,fig_style,ch, 'pairs', total_pairs)
plot_main(success_rates_race,fig_style,ch,'race', total_race)
plot_main(success_rates_Gender, fig_style,ch,'gender', total_Gender)   