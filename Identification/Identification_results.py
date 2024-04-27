from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import numpy as np
import argparse, pickle
from PIL import ImageFile
from os.path import isfile, join
from os import listdir, makedirs
ImageFile.LOAD_TRUNCATED_IMAGES = True

from deepface import DeepFace
from shutil import rmtree, copy
import pickle
import numpy as np
from PIL import Image
from os.path import join, isdir, isfile, basename
from os import listdir, remove, walk, mkdir
import os
import json, pickle
import requests, time
from random import shuffle, choice
import pandas as pd

parser = argparse.ArgumentParser()# Add an argument
parser.add_argument('--score_file', type=str, required=True, help='path to the score file')# Parse the argument

parser.add_argument('--fig_style', type=str, required=False, help='choose <seperate> to have a figure for each model', default='all')# Parse the argument

args = parser.parse_args()
scores_file = args.score_file
fig_style = args.fig_style
fontsize=11
term_dict= {'all': 'Scenario1_All', 'each': 'Scenario2_Each'}

# scores_file = 'Identification_Scores_CIAGAN_000016.pkl'
# fig_style = 'all'

with open(scores_file, 'rb') as f:
    scores= pickle.load(f)
    f.close()
              
# scores, ind = check_if_the_file_exist(scores_file)

colors = ['red', 'blue', 'yellow',  'green', 'pink', 'gold', 'purple', 'brown', ]
color_dic = {"white_males": colors[0], "White_Males": colors[0], "black_males": colors[1], "Black_Males": colors[1],  
             "asian_males": colors[2], "Asian_Males": colors[2], "indian_males": colors[3], "Indian_Males": colors[3], 
             
             "white_females": colors[4], "White_Females": colors[4], "indian_females": colors[5], "Indian_Females": colors[5],  
             "black_females": colors[6], "Black_Females": colors[6], "asian_females": colors[7], "Asian_Females": colors[7], 
             "white": colors[0], "White": colors[0], "Caucasian": colors[0], 'indian': colors[3], 'Indian': colors[3],
             'black': colors[1], 'Black': colors[1], 'African': colors[1], 'asian': colors[2], 'Asian': colors[2], 
             'males': colors[0], 'Males': colors[0], "females": colors[1], "Females": colors[1]} 

hatches=['/', '\\', '//', '-', '+', 'x', 'o', 'O', '.', '*']
hatch_dic = {"white_males": hatches[0], "White_Males": hatches[0], "black_males": hatches[1], "Black_Males": hatches[1],  
             "asian_males": hatches[2], "Asian_Males": hatches[2], "indian_males": hatches[3], "Indian_Males": hatches[3], 
             
             "white_females": hatches[4], "White_Females": hatches[4], "indian_females": hatches[5], "Indian_Females": hatches[5],  
             "black_females": hatches[6], "Black_Females": hatches[6], "asian_females": hatches[7], "Asian_Females": hatches[7], 
             "white": hatches[0], "White": hatches[0], "Caucasian": hatches[0], 'indian': hatches[3], 'Indian': hatches[3],
             'black': hatches[1], 'Black': hatches[1], 'African': hatches[1], 'asian': hatches[2], 'Asian': hatches[2], 
             'males': hatches[0], 'Males': hatches[0], "females": hatches[1], "Females": hatches[1]} 

line_styles=['-', '--', '-.', ':','dotted'  ]

def mean_score_calculator(rates, term, method):
        sum_all=0
        sum_rates=0
        means_dataset={}
        for dataset in rates.keys():
            sum_dataset=0
            sum_rates_dataset=0
            for demo in rates[dataset].keys():
                if type(rates[dataset][demo])==dict:
                    
                    score=rates[dataset][demo][term]['test']
                    if method!='Original':
                        score=100-score
                        
                    sum_rates_dataset+=score
                    sum_dataset+=1
                    
                    sum_rates+=score
                    sum_all+=1

            means_dataset[dataset]= sum_rates_dataset/sum_dataset
        means_total=sum_rates/sum_all
        return means_total, means_dataset


def sort_demos(total_demos, term):
    new_demos= {}
    for demo in total_demos.keys():
        if type(total_demos[demo])==dict:
            new_demos[demo]= total_demos[demo][term]['test']
    new_demos = sorted(new_demos.items(), key=lambda x:x[1],  reverse=False)
    new_demos = [d[0] for d in new_demos]

    return new_demos
    
def scores_filter(scores):
    methods = list(scores.keys())
    for method in methods:
        datasets= list(scores[method].keys())
        models= list(scores[method][datasets[0]].keys()) 
        for dataset in datasets:
            for model in models:
                if scores[method][dataset][model]['done']==0:
                    del scores[method][dataset][model]
    return scores    
import matplotlib.pyplot as plt    
scores=scores_filter(scores)    
#plotting three methods for each dataset
methods = list(scores.keys())
# methods= [m for m in methods if 'Org' not in m]
for method in methods:
    datasets= list(scores[method].keys())
    models= list(scores[method][datasets[0]].keys())
    for term in ["all", "each"]:
        if fig_style == 'all':
            
            num_rows =len(models)
            num_cols= 1
            fig, ax = plt.subplots(num_rows, num_cols, figsize=(5*2*num_cols, 5*num_rows))
            figname = "Identification-" + method + "-" + term+ '-'+ 'all_models' 
            k=0
            for model in models:
                k+=1
                plt.subplot(num_rows, num_cols, k)
                X_axis = np.arange(len(datasets))
                dataset_x=0
                scores_for_mean={}
                for dataset in datasets:
                    if scores[method][dataset][model]['done']==1:
                        scores_for_mean[dataset]=scores[method][dataset][model]


                        demo_num = len(list(scores[method][dataset][model].keys()))
                        Total_length = .9
                        width_demo = Total_length/(demo_num)
                        i = 1/2
                        sorted_demos=sort_demos(scores[method][dataset][model], term)
                        for demo in sorted_demos:
                            if method=='Original':
                                d=scores[method][dataset][model][demo][term]["test"]
                            else:
                                d=(100-scores[method][dataset][model][demo][term]["test"])
                            # print(d)
                            plt.bar(dataset_x -Total_length/2 + width_demo*i , d, width_demo, label = demo, color=color_dic[demo], hatch=hatch_dic[demo]) 
                            i+=1
                        dataset_x+=1            # plt.bar(X_axis -Total_length/2 + width_demo*i , scores[method][FR][dataset][demo]["each"]["test"], width_demo, label = demo + "-" + "each") 
                plt.xticks(X_axis, datasets, fontsize=fontsize) 
                plt.title(term)
                means_total, mean_dataset= mean_score_calculator(scores_for_mean, term, method)

                plt.axhline(y = means_total, color = colors[0], linestyle = line_styles[0], label = "Total: "+"{:.0f}".format(means_total)) 
                # trans = transforms.blended_transform_factory( ax.get_yticklabels()[0].get_transform(), ax.transData) 
                # plt.text(-1,means_total, "{:.0f}".format(means_total), color=colors[0],   ha="left", va="center")
                cl=0
                for dataset in mean_dataset.keys():
                    cl+=1
                    plt.axhline(y = mean_dataset[dataset], color = colors[cl], linestyle = line_styles[cl], label = dataset+": {:.0f}".format(mean_dataset[dataset])) 
                    # trans = transforms.blended_transform_factory( ax.get_yticklabels()[0].get_transform(), ax.transData) 
                    # plt.text(-1,mean_dataset[dataset], "{:.0f}".format(mean_dataset[dataset]), color=colors[cl],   ha="left", va="center")
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(
                    ), fontsize=fontsize-2, loc='center left', bbox_to_anchor=(1, 0.5))
        
                if method!='Original':
                    
                    plt.ylabel("Obfuscation Success Rates", fontsize=fontsize) 
                else:
                    plt.ylabel("Identification Rates", fontsize=fontsize) 
                # plt.title("Number of Students in each group") 
                # handles, labels = plt.gca().get_legend_handles_labels()
                # by_label = dict(zip(labels, handles))
                # # plt.legend(by_label.values(), by_label.keys(), fontsize=fontsize)
                # plt.legend(by_label.values(), by_label.keys(), fontsize=fontsize, loc='lower center', bbox_to_anchor=(.5, 1.04), ncol=5)
        
            # plt.legend(fontsize=fontsize) 
            # manager = plt.get_current_fig_manager()
            # manager.full_screen_toggle()
            plt.tight_layout()
            # plt.show()
            directory=join(term_dict[term], fig_style)
            if not isdir(directory):
                makedirs(directory)
            figname2=join(directory, figname)
            plt.savefig(figname2+'.jpg', dpi=150)
        else:

            for model in models:
                num_rows =1
                num_cols= 1
                fig, ax = plt.subplots(num_rows, num_cols, figsize=(10*num_cols, 5*num_rows))
                figname = "Identification-" + method + "-" + term+ '-'+ model 
                k=0
                k+=1
                plt.subplot(num_rows, num_cols, k)
                X_axis = np.arange(len(datasets))
                dataset_x=0
                scores_for_mean={}

                for dataset in datasets:
                    if scores[method][dataset][model]['done']==1:
                        scores_for_mean[dataset]=scores[method][dataset][model]

                        demo_num = len(list(scores[method][dataset][model].keys()))
                        Total_length = .9
                        width_demo = Total_length/(demo_num)
                        i = 1/2
                        sorted_demos=sort_demos(scores[method][dataset][model], term)
                        for demo in sorted_demos:
                   
                            if method=='Original':
                                d=scores[method][dataset][model][demo][term]["test"]
                            else:
                                d=(100-scores[method][dataset][model][demo][term]["test"])
                            # print(d)
                            plt.bar(dataset_x -Total_length/2 + width_demo*i , d, width_demo, label = demo, color=color_dic[demo], hatch=hatch_dic[demo]) 
                            i+=1
                        dataset_x+=1            # plt.bar(X_axis -Total_length/2 + width_demo*i , scores[method][FR][dataset][demo]["each"]["test"], width_demo, label = demo + "-" + "each") 
                plt.xticks(X_axis, datasets, fontsize=fontsize) 
                plt.title(term)
                means_total, mean_dataset= mean_score_calculator(scores_for_mean, term, method)

                plt.axhline(y = means_total, color = colors[0], linestyle = line_styles[0], label = "Total: "+"{:.0f}".format(means_total)) 
                # trans = transforms.blended_transform_factory( ax.get_yticklabels()[0].get_transform(), ax.transData) 
                # plt.text(-1,means_total, "{:.0f}".format(means_total), color=colors[0],   ha="left", va="center")
                cl=0
                for dataset in mean_dataset.keys():
                    cl+=1
                    plt.axhline(y = mean_dataset[dataset], color = colors[cl], linestyle = line_styles[cl], label = dataset+": {:.0f}".format(mean_dataset[dataset])) 
                    # trans = transforms.blended_transform_factory( ax.get_yticklabels()[0].get_transform(), ax.transData) 
                    # plt.text(-1,mean_dataset[dataset], "{:.0f}".format(mean_dataset[dataset]), color=colors[cl],   ha="left", va="center")
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(
                    ), fontsize=fontsize-2, loc='center left', bbox_to_anchor=(1, 0.5))  
                if method!='Original':
                    
                    plt.ylabel("Obfuscation Success Rates", fontsize=fontsize) 
                else:
                    plt.ylabel("Identification Rates", fontsize=fontsize) 

                # plt.title("Number of Students in each group") 
                # handles, labels = plt.gca().get_legend_handles_labels()
                # by_label = dict(zip(labels, handles))
                # # plt.legend(by_label.values(), by_label.keys(), fontsize=fontsize)
                # plt.legend(by_label.values(), by_label.keys(), fontsize=fontsize, loc='lower center', bbox_to_anchor=(.5, 1.03), ncol=5)
        
                # plt.legend(fontsize=fontsize) 
                # manager = plt.get_current_fig_manager()
                # manager.full_screen_toggle()
                plt.tight_layout()
                # plt.show()
                directory=join(term_dict[term], fig_style, method)
                if not isdir(directory):
                    makedirs(directory)
                figname2=join(directory, figname)
                plt.gcf().set_size_inches(10*num_cols, 5*num_rows)

                plt.savefig(figname2+'.jpg', dpi=150)


plt.close('all')  
import seaborn as sns               
#disparity maps
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
my_colors=['red', 'green']
my_cmap= ListedColormap(my_colors)
bounds = [0, .8,  1]
my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))
patterns = ['', 'oo', '////', 'XXX', '*']
fairness_thresh=.8



def plot_demographic_parity(identification, obf_method, term):
    datasets = list(identification.keys())
    models= list(identification[datasets[0]].keys())

    if fig_style=='all':

        num_cols= len(datasets)
        num_rows =len(models)
        fig, ax = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
        figname = "Bias-Identification-" + obf_method + "-" + term + '-all_models' 

        d=1
        for model in models:
            biases={}
            
            # datasets = list(identification[model].keys())
            # num_cols= len(datasets)
            # num_rows= int(np.ceil(len(datasets)/2))

            for dataset in datasets:

                if identification[dataset][model]['done']==1:

                    biases[dataset]={}
                    
           
                    demos=sort_demos(identification[dataset][model], term)
    
            
                    num_demos =len(demos)
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
                                if method=='Original':
                                    biases[dataset][demo][demo2]=(identification[dataset][model][demo][term]['test'])/(identification[dataset][model][demo2][term]['test'])
                                else:
                                    biases[dataset][demo][demo2]=(100-identification[dataset][model][demo][term]['test'])/(100-identification[dataset][model][demo2][term]['test'])
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
                    plt.subplot(num_rows, num_cols, d)
                    plt.title(model + '-' + dataset)
            
                        
                    d+=1
                    sns.heatmap(dataset_biass,  xticklabels=Demos, yticklabels=Demos,  mask= matrix,  cmap=my_cmap,  norm=my_norm) 
        # plt.show()
        plt.tight_layout()
        directory=join(term_dict[term], fig_style)
        if not isdir(directory):
            makedirs(directory)
        figname2=join(directory, figname)
        plt.savefig(figname2+'.jpg', dpi=150)
                    
    else:
            
        for model in models:
            figname = "Bias-Identification-" + obf_method + "-" + term + '-' + model


            biases={}
            d=1
            num_rows = 1
            # num_rows= int(np.ceil(len(datasets)/2))
            num_cols=len(datasets)
            fig, ax = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))

            for dataset in datasets:
                if identification[dataset][model]['done']==1:
    
    
                    biases[dataset]={}
                    
           
                    demos=sort_demos(identification[dataset][model], term)
    
                    num_demos =len(demos)
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
                               if method=='Original':
                                   biases[dataset][demo][demo2]=(identification[dataset][model][demo][term]['test'])/(identification[dataset][model][demo2][term]['test'])
                               else:
                                   biases[dataset][demo][demo2]=(100-identification[dataset][model][demo][term]['test'])/(100-identification[dataset][model][demo2][term]['test'])
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
                    plt.subplot(num_rows,num_cols, d)
                    plt.title(dataset)
            
                        
                    d+=1
                    sns.heatmap(dataset_biass,  xticklabels=Demos, yticklabels=Demos,  mask= matrix,  cmap=my_cmap,  norm=my_norm) 
            # plt.show()
            plt.tight_layout()
            directory=join(term_dict[term], fig_style, method)
            if not isdir(directory):
                makedirs(directory)
            figname2=join(directory, figname)
            
            plt.savefig(figname2+'.jpg', dpi=150)

for obf_method in scores.keys():
    for term in ['all', 'each']:
        plot_demographic_parity(scores[obf_method],obf_method, term)
   